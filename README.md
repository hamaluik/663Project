# Kenton's MecE 663 Project
This is a custom deal.ii application that has been developed to solve a heat transfer problem. While the
problem would be extremely difficult to solve analytically due to the "complex" boundary conditions, the
solved problem is quite simple for a finite-element analysis.

## Problem Domain
The problem is a small rectangle with dimensions 2.5cm x 1cm, and is composed of 3 distinct boundary conditions.
On y = 1, from x = 1 to x = 2.5, a constant temperature of 400K is applied. On x = 0, from y = 0 to y = 1,
a heat flux in the x direction only that is applied according to the equation: B(1-y) where B is a constant.

There is a heat generation term in the given PDE, equal to Ax^2 where A is a constant.

## Default Constants
	Kx = 15 W/m^2
	Ky = 25 W/m^2
	A is in (0, 5) x 1000 W/m^5
	B is 50 W/m^3
	T (temperature at the Dirichlet boundary) = 400K

## Implementation 
The program is written with a static geometry and boundary locations / types, however the above constants
may be changed through the command-line (as well as several other options pertaining to the finite-element
formulation itself.

### Command-line
The program may be invoked and produce a solution using the following syntax:
	./project [OPTIONS] [SOLUTION_FILE.vtk]
Where both [OPTIONS] and [SOLUTION_FILE.vtk] are optional parameters. [OPTIONS] follows unix-style command-line
options, with the following values being accepted:
	-?		shows a help menu
	-r <num>	refines the mesh <num> times (defaults to 0)
	-f		write the generated mesh to file for visualization
	-s		actually solve the problem
	-q <num|auto>	allows you to manually specify the number of quadrature points used (defaults to auto)
	-i <num>	set the maximum number of CG iterations performed for the solution (defaults to 1000)
	-t <tol>	set the tolerance for the residual in the CG solution (defaults to 1e-12)
	-o <order>	set the Lagrange interpolation polynomial order (defaults to 1)
	-x <Kx>		set the Kx value (defaults to 15)
	-y <Ky>		set the Ky value (defaults to 25)
	-A <A>		set the A value (defaults to 5000)
	-B <B>		set the B value (defaults to 50)
	-T <temp>	set the boundary temperature at the Dirichlet boundary (defaults to 400)

If [SOLUTION_FILE.vtk] is specified, the solution will be graphically written to the specified file (in VTK
format). If it **is not** specified and a solution is performed, the solution will be written to "solution.vtk",
overwriting that file.
