// all our includes
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
#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <map>
#include <ctime>

// make sure we're in the deal.ii namespace
using namespace dealii;

// the main application lives here
class Project {
public:
	// initialize the constructor
	Project() : fe(1), dofHandler(triangulation) {}
	// the way to run the program!
	void run(int argc, char **argv);

private:
	// functions that define the system
	// different for different types of problems
	// (ie thermal problems vs. structural problems)
	double systemValue(FEValues<2> &feValues, unsigned int i, unsigned int j, unsigned int q);
	double rhsValue(FEValues<2> &feValues, unsigned int i, unsigned int q);

	// helper functions
	void mesh(int refinements, bool toFile);
	void setup();
	void assemble(int quadPoints, unsigned int boundary);
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

// our application entry point
int main(int argc, char *argv[]) {
	Project project;
	project.run(argc, argv);
	return 0;
}

// define the system portion of the weak formulation of the PDE here
// (use quadrature points)
double Project::systemValue(FEValues<2> &feValues, unsigned int i, unsigned int j, unsigned int q) {
	return (feValues.shape_grad(i, q) * feValues.shape_grad(j, q) * feValues.JxW(q));
}

// define the RHS portion of the weak formulation of the PDE here
// (use quadrature points)
double Project::rhsValue(FEValues<2> &feValues, unsigned int i, unsigned int q) {
	return (feValues.shape_value(i, q) * feValues.JxW(q));
}

// run the calculations
void Project::run(int argc, char *argv[]) {
	// default options
	int refinements = 0;
	bool meshToFile = true;
	bool solveProblem = true;
	int quadPoints = 2;
	unsigned int nIterations = 1000;
	double tolerance = 1e-12;
	unsigned int boundary = 3;

	// parse the command line
	// skip the first argument
	int i = 1;
	while(i < argc) {
		// check for mesh refinement level
		if(strcmp(argv[i], "-r") == 0) {
			i++;
			if(i >= argc) {
				std::cout << "ERROR: Invalid option to parameter: -r" << std::endl;
			}
			else {
				refinements = atoi(argv[i++]);
			}
		}
		// check for outputting the mesh to file
		else if(strcmp(argv[i++], "-smo") == 0) {
			meshToFile = false;
		}
		// check to see if we want to actually solve it
		else if(strcmp(argv[i++], "-dns") == 0) {
			solveProblem = false;
		}
		// check for mesh refinement level
		else if(strcmp(argv[i], "-qp") == 0) {
			i++;
			if(i >= argc) {
				std::cout << "ERROR: Invalid option to parameter: -q" << std::endl;
			}
			else {
				quadPoints = atoi(argv[i++]);
			}
		}
		// check for which boundary to apply at
		/*else if(strcmp(argv[i], "-b") == 0) {
			i++;
			if(i >= argc) {
				std::cout << "ERROR: Invalid option to parameter: -b" << std::endl;
			}
			else {
				boundary = atoi(argv[i++]);
			}
		}*/
		// check for help
		else if(strcmp(argv[i], "-?") == 0) {
			// print some help!
			std::cout << "Kenton's 663 Project - help" << std::endl;
			std::cout << "Command line options:" << std::endl;
			std::cout << "\t-r <refinements>\tset the number of mesh refinement levels" << std::endl;
			std::cout << "\t-smo\tsuppress output of the mesh to file (eps)" << std::endl;
			std::cout << "\t-dns\tdo not solve the problem, compute the mesh only" << std::endl;
			std::cout << "\t-qp <num points>\tmanually define the number of gauss quadrature points to use" << std::endl;

			// and stop
			return;
		}
		// we don't know what this is!
		else {
			std::cout << "ERROR: unknown option: " << argv[i] << std::endl;
			// stop, we hit a roadblock
			return;
		}
	}

	// echo the options back to the user
	std::cout << "-- Options --" << std::endl;
	std::cout << "Mesh refinement set to: " << refinements << std::endl;
	std::cout << "Write mesh to file: " << (meshToFile ? "yes" : "no") << std::endl;
	std::cout << "Perform solution: " << (solveProblem ? "yes" : "no") << std::endl;
	std::cout << "Number of Gauss quadrature points: " << quadPoints << std::endl;
	std::cout << "Maximum number of solution iterations: " << nIterations << std::endl;
	std::cout << "Minimum normed residual tolerance: " << tolerance << std::endl;

	// suppress deal.ii solver output
	//deallog.depth_console(0);

	// run the various steps
	std::cout << "-- Preprocessing --" << std::endl;
	std::cout << "Meshing domain... ";
	mesh(refinements, meshToFile);
	std::cout << "done!" << std::endl;
	std::cout << "Setting up the system... ";
	setup();
	std::cout << "done!" << std::endl;
	std::cout << "Assembling the system matrix and RHS... ";
	assemble(quadPoints, boundary);
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
	// define the geometry
	// read it in from file
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
	//cell->face(0)->set_boundary_indicator(2);
	//cell->face(2)->set_boundary_indicator(2);
	//cell->face(3)->set_boundary_indicator(1);
	cell->face(0)->set_boundary_indicator(1);
	cell->face(1)->set_boundary_indicator(2);
	cell->face(2)->set_boundary_indicator(3);
	cell->face(3)->set_boundary_indicator(4);
	// now the second cell
	++cell;
	//cell->face(0)->set_boundary_indicator(2);
	//cell->face(1)->set_boundary_indicator(2);
	//cell->face(2)->set_boundary_indicator(3);

	// perform refinements
	triangulation.refine_global(refinements);

	// write it to file
	if(toFile) {
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
void Project::assemble(int quadPoints, unsigned int boundary) {
	// use Gauss quadrature to solve the problem
	// with a specified number of points given as an argument
	QGauss<2> quadrature(quadPoints);

	// ?
	// use this to make sure only what we need gets updated between cells
	FEValues<2> feValues(fe, quadrature, update_values | update_gradients | update_JxW_values);

	// a couple of shortcuts for helping to define the system
	const unsigned int dofsPerCell = fe.dofs_per_cell;
	const unsigned int nQPoints = quadrature.size();

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
		// reset the geometry for the cell
		feValues.reinit(cell);

		// reset the local and RHS formulations
		localMatrix = 0;
		localRHS = 0;

		// now loop over each component of the local system matrix
		for(unsigned int i = 0; i < dofsPerCell; ++i) {
			// deal with the system matrix
			for(unsigned int j = 0; j < dofsPerCell; ++j) {
				// now loop over the quadrature points
				for(unsigned int q = 0; q < nQPoints; ++q) {
					// and add the local contribution
					localMatrix(i, j) += systemValue(feValues, i, j, q);
				}
			}

			// deal with the RHS
			// again loop over the quadrature points
			for(unsigned int q = 0; q < nQPoints; ++q) {
				localRHS(i) += rhsValue(feValues, i, q);
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

	// ok, the static system and RHS formulations are complete
	// next, fill in the boundary conditions
	// get the boundary values associated with specific DOFs
	std::map<unsigned int, double> boundaryValues;
	// apply the boundary values
	VectorTools::interpolate_boundary_values(dofHandler, boundary, ZeroFunction<2>(), boundaryValues);
	//VectorTools::interpolate_boundary_values(dofHandler, 1, ZeroFunction<2>(), boundaryValues);
	//VectorTools::interpolate_boundary_values(dofHandler, 2, ZeroFunction<2>(), boundaryValues);
	//VectorTools::interpolate_boundary_values(dofHandler, 3, ZeroFunction<2>(), boundaryValues);
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
	SolverCG<> solver(solverControl);

	// measure solution time
	startTime = time(0);

	// and solve that puppy!
	// use an identity preconditioner
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

	// attach some data to those DOFs
	dataOut.add_data_vector(solution, "solution");

	// transform the data into an intermediate format
	dataOut.build_patches();

	// and write it to file
	std::ofstream output("solution.vtk");
	dataOut.write_vtk(output);
}

// print out stats about the solution
void Project::postStats() {
	double solveTime = difftime(endTime, startTime);
   std::cout << "Total solution time (s): "
             << solveTime << std::endl
             << "Iterations needed to obtain convergence: "
             << solutionIterations << std::endl;
}
