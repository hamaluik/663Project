// deal.II includes
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vectors.h>
#include <deal.II/numerics/matrices.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/grid_out.h>

// C includes for getopt
#include <stdio.h>
#include <stdlib.h>

// regular c++ includes
#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <map>
#include <ctime>
#include <cstdlib>
#include <string>

// make sure we're in the deal.ii namespace
using namespace dealii;

// the main application lives here
class Project {
public:
	// initialize the constructor
	Project(int order, double _Kx, double _Ky, double _A, double _B, double _T, char *_solutionFileName) : fe(order), dofHandler(triangulation) {
		Kx = _Kx;
		Ky = _Ky;
		A = _A;
		B = _B;
		T = _T;
		solutionFileName = _solutionFileName;
	}
	// the way to run the program!
	void run(int refinements, bool meshToFile, bool solveProblem, int quadPoints, unsigned int nIterations, double tolerance);

private:
	// problem constants
	double Kx, Ky, A, B, T;

	// misc. options
	char *solutionFileName;

	// functions that define the system
	// different for different types of problems
	// (ie thermal problems vs. structural problems)
	double systemValue(FEValues<2> &feValues, unsigned int i, unsigned int j, unsigned int q);
	double rhsValue(FEValues<2> &feValues, unsigned int i, unsigned int q);
	double boundaryRHS(FEFaceValues<2> &feFaceValues, int boundary, unsigned int i, unsigned int q);

	// helper functions
	void mesh(int refinements, bool toFile);
	void setup();
	void assemble(int quadPoints);
	void preStats();
	void solve(unsigned int n, double tolerance);
	void results();
	void postStats();

	// internal containers
	Triangulation<2> triangulation;
	FE_Q<2> fe;
	DoFHandler<2> dofHandler;
	SparsityPattern sparsityPattern;
	SparseMatrix<double> systemMatrix;
	Vector<double> solution;
	Vector<double> systemRHS;

	// internal states
	time_t startTime, endTime;
	unsigned int solutionIterations;
};

// *********************************************************************************//
// ***** define the problem-specific system, RHS, and boundary conditions here *****//
// *********************************************************************************//

// define the system matrix portion of the weak formulation of the PDE here
// (use quadrature points)
double Project::systemValue(FEValues<2> &feValues, unsigned int i, unsigned int j, unsigned int q) {
	// ((d/dx)phi_i * Kx * (d/dx)phi_j) + ((d/dy)phi_i * Ky * (d/dy)phi_j)
	return	(feValues.shape_grad(i, q)[0] * Kx * feValues.shape_grad(j, q)[0] * feValues.JxW(q))
	+	(feValues.shape_grad(i, q)[1] * Ky * feValues.shape_grad(j, q)[1] * feValues.JxW(q));
}

// define the RHS portion of the weak formulation of the PDE here
// (use quadrature points)
double Project::rhsValue(FEValues<2> &feValues, unsigned int i, unsigned int q) {
	// separate this out for readability and debugging purposes
	// let the compiler optimize it
	double x = feValues.quadrature_point(q)[0];
	double x2 = pow(x, 2);
	// phi_i * A * x^2
	return (feValues.shape_value(i, q) * A * x2 * feValues.JxW(q));
}

// define the RHS contribution for boundaries
double Project::boundaryRHS(FEFaceValues<2> &feFaceValues, int boundary, unsigned int i, unsigned int q) {
	// deal with Neumann BC here!
	double value = 0;
	if(boundary == 1) {
		// - phi_i * B * (1 - y)
		// note: not dealing with normal vectors here
		// since we know exactly where the boundary will always be
		// and so its norm is a constant and can be simplified out
		value = -1 * feFaceValues.shape_value(i, q) * B * (1 - feFaceValues.quadrature_point(q)[1]) * feFaceValues.JxW(q);
	}
	else if(boundary == 2) {
		// phi_i * 0
		value = 0;
	}
	return value;
}

// define functions for boundary values
// (constant Diritchlet BC)
template<int dim>
class DirichletConst:public Function<dim> {
private:
	double V;
public:
	// return a configurable constant value
	DirichletConst(double _V): Function<dim>() {
		V = _V;
	}
	// ignore warnings about unused p & component parameters
	#pragma GCC diagnostic ignored "-Wunused-parameter"
	double value(const Point<dim> &p, const unsigned int component = 0) const {
		return V;
	}
};

// run the calculations
void Project::run(int refinements, bool meshToFile, bool solveProblem, int quadPoints, unsigned int nIterations, double tolerance) {
	// echo the options back to the user
	std::cout << "-- Options --" << std::endl;
	std::cout << "Mesh refinement set to: " << refinements << std::endl;
	std::cout << "Write mesh to file: " << (meshToFile ? "yes" : "no") << std::endl;
	std::cout << "Perform solution: " << (solveProblem ? "yes" : "no") << std::endl;
	std::cout << "Number of Gauss quadrature points: " << quadPoints << std::endl;
	std::cout << "Maximum number of solution iterations: " << nIterations << std::endl;
	std::cout << "Minimum normed residual tolerance: " << tolerance << std::endl;
	if(solveProblem) std::cout << "Writing solution to file: " << solutionFileName << std::endl;

	// echo back problem constants
	std::cout << "-- Constants --" << std::endl;
	std::cout << "Kx = " << Kx << " (W/mK)" << std::endl;
	std::cout << "Ky = " << Ky << " (W/mK)" << std::endl;
	std::cout << "A  = " << A << " (W/m^5)" << std::endl;
	std::cout << "B  = " << B << " (W/m^3)" << std::endl;
	std::cout << "T  = " << T << " (K)" << std::endl;

	// suppress deal.ii solver output
	//deallog.depth_console(0);

	// run the various steps
	std::cout << "-- Preprocessing --" << std::endl;
	std::cout << "Meshing domain... ";
	mesh(refinements, meshToFile);
	std::cout << "done!" << std::endl;
	// set the system up for solution
	std::cout << "Setting up the system... ";
	setup();
	std::cout << "done!" << std::endl;
	std::cout << "Assembling the system matrix and RHS... ";
	assemble(quadPoints);
	std::cout << "done!" << std::endl;

	// output some info about the mesh
	std::cout << "-- Mesh Stats --" << std::endl;
	preStats();

	// return early if we don't want to solve
	if(!solveProblem) {
		return;
	}

	// solve the system
	std::cout << "-- Solution --" << std::endl;
	std::cout << "Solving system... " << std::endl;
	solve(nIterations, tolerance);
	std::cout << "done!" << std::endl;
	std::cout << "Outputting solution results... ";
	results();
	std::cout << "done!" << std::endl;

	// write some stats about the solution
	std::cout << "-- Solution Stats --" << std::endl;
	postStats();
}

// generate a mesh of the domain
void Project::mesh(int refinements, bool toFile) {
	// TODO: create triangulation solely in code
	// for this single problem (eliminate need for .ucd file!)

	// define the geometry
	// read it in from file (for now)
	GridIn<2> gridIn;
	gridIn.attach_triangulation(triangulation);
	std::ifstream inFile("project.ucd");
	gridIn.read_ucd(inFile);

	// set up the boundaries
	Triangulation<2>::active_cell_iterator cell = triangulation.begin_active();
	// for the first cell
	// face 0 = left
	// face 1 = right
	// face 2 = bottom
	// face 3 = top
	cell->face(0)->set_boundary_indicator(1); // dO1
	cell->face(2)->set_boundary_indicator(2); // dO2
	cell->face(3)->set_boundary_indicator(2); // dO2
	// now the second cell
	++cell;
	cell->face(2)->set_boundary_indicator(2); // dO2
	cell->face(1)->set_boundary_indicator(2); // dO2
	cell->face(3)->set_boundary_indicator(3); // dO3

	// perform refinements
	triangulation.refine_global(refinements);

	// write it to file
	if(toFile) {
		// save as an eps for viewing
		std::ofstream out("mesh.eps");
		GridOut gridOut;
		gridOut.write_eps(triangulation, out);
	}
}

// prepare the matrices for filling
void Project::setup() {
	// fill in the degrees of freedom
	dofHandler.distribute_dofs(fe);

	// deal with the sparsity in the matrix
	CompressedSparsityPattern compressedSparsity(dofHandler.n_dofs());
	DoFTools::make_sparsity_pattern(dofHandler, compressedSparsity);
	sparsityPattern.copy_from(compressedSparsity);

	// now create the system matrix from the sorted sparsity pattern
	systemMatrix.reinit(sparsityPattern);

	// and the solution and RHS vectors
	solution.reinit(dofHandler.n_dofs());
	systemRHS.reinit(dofHandler.n_dofs());
}

// assemble the system matrix and RHS vector
void Project::assemble(int quadPoints) {
	// use Gauss quadrature to solve the problem
	// with a specified number of points given as an argument
	QGauss<2> quadrature(quadPoints);
	// use "faceQuadrature" to deal with Neumann BC
	// NOTE: since it is a boundary, it is of a lower dimension
	// i.e., it's a line, so it's in 1D space
	QGauss<1> faceQuadrature(quadPoints);

	// use FEValues and FEFaceValues to calculate values and gradients of
	// shape functions and coordinates at each quadrature point
	// use the flags to make sure only what we need gets updated between cells
	// we will be needing the shape function values, gradients, the jacobians, and the quadrature values
	FEValues<2> feValues(fe, quadrature, update_values | update_gradients | update_JxW_values | update_quadrature_points);
	FEFaceValues<2> feFaceValues(fe, faceQuadrature, update_values | update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors);

	// a couple of shortcuts for helping to define the system
	// the total number of degrees of freedom per cell
	const unsigned int dofsPerCell = fe.dofs_per_cell;
	// number of quadrature points on the cells
	const unsigned int nQPoints = quadrature.size();
	// number of quadrature points on the boundaries
	const unsigned int nFaceQPoints = faceQuadrature.size();

	// initialize the local element and RHS formulations
	FullMatrix<double> localMatrix(dofsPerCell, dofsPerCell);
	Vector<double> localRHS(dofsPerCell);

	// temporary array to hold local DOF indices
	std::vector<unsigned int> localDOFIndices(dofsPerCell);

	// now loop over all the cells in the mesh
	// create iterators for the start and end of the active cell list
	DoFHandler<2>::active_cell_iterator cell = dofHandler.begin_active();
	DoFHandler<2>::active_cell_iterator endCell = dofHandler.end();
	// now actually loop through all the cells
	for(; cell != endCell; ++cell) {
		// reset the FE values information for this cell
		feValues.reinit(cell);

		// reset the local and RHS formulations
		localMatrix = 0;
		localRHS = 0;

		// now loop over each component of the local system matrix (and RHS entries)
		// loop over the quadrature points
		for(unsigned int q = 0; q < nQPoints; ++q) {
			// loop over "i" indices
			for(unsigned int i = 0; i < dofsPerCell; ++i) {
				// loop over "j" indices
				for(unsigned int j = 0; j < dofsPerCell; ++j) {
					// deal with the system matrix
					// and add the local contribution
					localMatrix(i, j) += systemValue(feValues, i, j, q);
				}

				// deal with the RHS
				localRHS(i) += rhsValue(feValues, i, q);
			}
		}

		// deal with boundaries
		// loop over all the faces on the cell
		for(unsigned int face = 0; face<GeometryInfo<2>::faces_per_cell; ++face) {
			// see if this face is on a boundary
			if(cell->face(face)->at_boundary()) {
				// yup, on a boundary
				// store the boundary for easy access
				int boundary = cell->face(face)->boundary_indicator();
				// make sure it's one of ours
				// (note: we have "Neumann" conditions on BC 1 and 2
				// the Dirichlet BC on BC 3 will be dealt with later
				if(boundary == 1 || boundary == 2) {
					// boundary 1, heat transfer (q)
					// reinit the face values
					feFaceValues.reinit(cell, face);
					// loop over all the quadrature points
					for(unsigned int q = 0; q < nFaceQPoints; ++q) {
						// and loop over all the DOFS in the cell
						for(unsigned int i = 0; i < dofsPerCell; ++i) {
							// apply the Neumann BC that is on this boundary
							localRHS(i) += boundaryRHS(feFaceValues, boundary, i, q);
						}
					}
				}
			}
		}

		// transfer the local cell into the global matrix
		// but first translate the DOF indices for this cell
		cell->get_dof_indices(localDOFIndices);
		// now loop over all the DOF in the cell
		for(unsigned int i = 0; i < dofsPerCell; ++i) {
			for(unsigned int j = 0; j < dofsPerCell; ++j) {
				// add to the system matrix
				systemMatrix.add(localDOFIndices[i], localDOFIndices[j], localMatrix(i, j));
			}
			// add to the RHS vector
			systemRHS(localDOFIndices[i]) += localRHS(i);
		}
	}

	// ok, the static system and RHS formulations (including Neumann conditions) are complete
	// next, fill in the Dirichlet boundary conditions
	// get the boundary values associated with specific DOFs
	std::map<unsigned int, double> boundaryValues;
	// apply the Diritchlet boundary values
	// boundary 3, const temperature
	VectorTools::interpolate_boundary_values(dofHandler, 3, DirichletConst<2>(T), boundaryValues);
	// and apply them to system and RHS formulations
	MatrixTools::apply_boundary_values(boundaryValues, systemMatrix, solution, systemRHS);
}

// print out stats about the mesh and problem before solving it
void Project::preStats() {
   std::cout << "Number of active cells: "
             << triangulation.n_active_cells()
             << std::endl
             << "Total number of cells: "
             << triangulation.n_cells()
             << std::endl
             << "Number of degrees of freedom: "
             << dofHandler.n_dofs()
             << std::endl;
}

// solve the system
void Project::solve(unsigned int n, double tolerance) {
	// control the CG solution
	SolverControl solverControl(n, tolerance);

	// create the solver
	// TODO: other solvers?
	SolverCG<> solver(solverControl);

	// measure solution time
	startTime = time(0);

	// and solve that puppy!
	// use an identity preconditioner (nothing fancy necessary for this relatively simple problem)
	solver.solve(systemMatrix, solution, systemRHS, PreconditionIdentity());

	// various performance metrics
	endTime = time(0);
	solutionIterations = solverControl.last_step() + 1;
}

// save results to file
void Project::results() {
	// deal with our data
	DataOut<2> dataOut;

	// get information from the DOFs
	dataOut.attach_dof_handler(dofHandler);

	// attach solution data to those DOFs
	dataOut.add_data_vector(solution, "Temperature");

	// transform the data into an intermediate format
	dataOut.build_patches();

	// and write it to file
	// create a file
	std::ofstream output(solutionFileName);

	// save the file for visualization
	dataOut.write_vtk(output);
}

// print out stats about the solution
void Project::postStats() {
	double solveTime = difftime(endTime, startTime);
   std::cout << "Total solution time (s): "
             << solveTime << std::endl
             << "CG iterations needed to obtain convergence: "
             << solutionIterations << std::endl;
}

// our application entry point
int main(int argc, char *argv[]) {
	// default options
	int refinements = 0;
	bool meshToFile = false;
	bool solveProblem = false;
	int quadPoints = -1;
	unsigned int nIterations = 1000;
	double tolerance = 1e-12;
	std::string solutionFile("solution.vtk");

	// values / constants in the problem
	int order = 1;
	double Kx = 15;
	double Ky = 25;
	double A = 5000;
	double B = 50;
	double T = 400;

	// parse the command line
	int opt;
	while((opt = getopt(argc, argv, "h?r:fsq:i:t:o:x:y:A:B:T:")) != -1) {
		switch(opt) {
		case 'h':
		case '?':
		{
			// help menu
			using namespace std;
			cout << "Usage: " << argv[0] << " [OPTIONS] [SOLUTION.vtk]" << endl;
			cout << "-- Options (defaults) --" << endl;
			cout << "\t-?\t\tshow this help menu" << endl;
			cout << "\t-r <level>\trefine the mesh <level> times (0)" << endl;
			cout << "\t-f\t\twrite the generated mesh to file for visualization (false)" << endl;
			cout << "\t-s\t\tsolve the problem (false)" << endl;
			cout << "\t-q <num|auto>\tuse <num> quadrature points for solution (or auto-calculate based on order) (auto)" << endl;
			cout << "\t-i <num>\tset the maximum number of CG iterations for solution (1000)" << endl;
			cout << "\t-t <tol>\tset the tolerance for the residual (1e-12)" << endl;
			cout << "\t-o <order>\tset the Lagrange polynomial order to <order> (1)" << endl;
			cout << "\t-x <Kx>\t\tset the Kx value (15)" << endl;
			cout << "\t-y <Ky>\t\tset the Ky value (25)" << endl;
			cout << "\t-A <A>\t\tset the A value (5000)" << endl;
			cout << "\t-B <B>\t\tset the B value (50)" << endl;
			cout << "\t-T <temp>\tset the temperature at boundary 3 (400)" << endl;
			return 0;
		}
		// set refinement level
		case 'r':
			refinements = atoi(optarg);
			break;
		// write mesh to an eps file for viewing?
		case 'f':
			meshToFile = true;
			break;
		// should we actually go ahead and solve it?
		case 's':
			solveProblem = true;
			break;
		// manually select the number of quadrature points
		case 'q':
			if(strcmp(optarg, "auto") == 0) {
				// if we need to auto-calculate the quadrature
				// points, set it to -1 so we know later
				quadPoints = -1;
			}
			else {
				quadPoints = atoi(optarg);
			}
			break;
		// maximum number of CG iterations before things fail
		case 'i':
			nIterations = atoi(optarg);
			break;
		// minimum CG residual tolerance before things stop
		case 't':
			tolerance = (double)atof(optarg);
			break;
		// Lagrange polynomial order
		case 'o':
			order = atoi(optarg);
			break;
		// Kx constant
		case 'x':
			Kx = (double)atof(optarg);
			break;
		// Ky constant
		case 'y':
			Ky = (double)atof(optarg);
			break;
		// A constant
		case 'A':
			A = (double)atof(optarg);
			break;
		// B constant
		case 'B':
			B = (double)atof(optarg);
			break;
		// Temperature constant (along dO3)
		case 'T':
			T = (double)atof(optarg);
			break;
		}
	}

	// deal with arguments
	// only a single [optional] argument - the solution file name
	if(optind < argc) {
		solutionFile = argv[optind];
	}

	// auto-calculate quadrature points
	if(quadPoints == -1) {
		// num points = (1/2)*(max order + 1)
		// for quadratic polynomials we have:
		//   (x^2+xy+y^2)' * x^2 * (x^2 + xy + y^2)'
		// = (x + y) * x^2 + (x * y)
		// => x^4
		// would need (1/2)*(4 + 1) = 2.5 => 3
		quadPoints = (int)ceil(0.5 * (double)(2 * (order - 1) + 2 + 1));
	}

	// create our project object
	Project project(order, Kx, Ky, A, B, T, (char *)solutionFile.c_str());

	// and run it!
	project.run(refinements, meshToFile, solveProblem, quadPoints, nIterations, tolerance);
	return 0;
}
