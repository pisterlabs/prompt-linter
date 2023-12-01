import openai
import os


prompt_template = """
You are an agent generating Python code for a CAD system. You are given:

(1) an objective that you are trying to achieve
(2) existing code that you can use as a starting point

You can use these functions:

Point3d(x,y,z) - returns a Point3d with coordinates x,y,z
Circle(Point3d,r) - create a circle at Point3d with radius r
Plane(Point3d,Vector3d) - create a plane at Point3d with normal Vector3d
Arc(Point3d,Point3d,Point3d) - create an arc from 3 points
Line(Point3d,Point3d) - create a line from 2 points
Brep.CreateFromLoft(CurveList,Point3d,Point3d,LoftType,Boolean) - create a Brep from a list of curves
Brep.CreateFromSweep(Curve, Curve, Boolean, Double) - create a Brep from a curve and a rail
Brep.CreateFromBox(Plane, Double, Double, Double) - create a Brep from a plane and 3 dimensions
Brep.CreateFromCylinder(Plane, Double, Double, Double) - create a Brep from a plane, radius, height, and cap type
Brep.CreateFromSphere(Plane, Double) - create a Brep from a plane and radius
Brep.CreateFromTorus(Plane, Double, Double) - create a Brep from a plane, major radius, and minor radius
Brep.CreateFromCone(Plane, Double, Double, Double) - create a Brep from a plane, radius, height, and cap type
Brep.CreateFromRevSurface(Curve, Plane, Double, Double) - create a Brep from a curve, plane, start angle, and end angle
Brep.CreateFromSurface(Surface) - create a Brep from a surface
Brep.CreateFromMesh(Mesh, Boolean) - create a Brep from a mesh
Sphere(Point3d,Double) - create a Sphere at Point3d with radius Double
Cylinder(Point3d,Vector3d,Double,Double) - create a Cylinder at Point3d with axis Vector3d, radius Double, and height Double
Cylinder(Point3d,Double,Double) - create a Cylinder at Point3d with radius Double and height Double

When you are done, please use these functions to write the model to a file:

File3dm() - create a new File3dm model
File3dm.AddPoint(Point3d) - add a Point3d to the File3dm
File3dm.AddPoint(Double, Double, Double) - add a Point3d to the File3dm using x,y,z coordinates
File3dm.AddLine(Point3d, Point3d) - add a Line to the File3dm
File3dm.AddSphere(Sphere) - add a Sphere to the File3dm
File3dm.AddCircle(Circle) - add a Circle to the File3dm
File3dm.AddPlane(Plane) - add a Plane to the File3dm
File3dm.AddArc(Arc) - add a curve object to the File3dm representing an Arc
File3dm.AddEllipse(Ellipse) - add a curve object to the File3dm representing an Ellipse
File3dm.AddSurface(Surface) - add a Surface to the File3dm
File3dm.AddExtrusion(Extrusion) - add an Extrusion to the File3dm
File3dm.AddMesh(Mesh) - add a Mesh to the File3dm
File3dm.AddLine(Line) - add a Line to the File3dm
File3dm.AddBrep(Brep) - add a Brep to the File3dm
File3dm.AddCylinder(Cylinder) - add a Cylinder to the File3dm

Point3d is a point in 3D space.

Based on your given objective, generate code using the functions above that you believe will generate code that represents the desired shapes. If you need to use any Python libraries, please import them.

Here are some examples of code:

EXAMPLE 1
=================================
CURRENT CODE:
---------------------------------
from rhino3dm import *
---------------------------------
OBJECTIVE: Generate a sphere at (1,2,3) with radius 10
YOUR CODE:
from rhino3dm import *
sph = Sphere(Point3d(1,2,3),10)

model = File3dm()
model.Objects.AddSphere(sph)

model.Write("export.3dm", 6)
=================================

EXAMPLE 2
=================================
CURRENT CODE:
---------------------------------
from rhino3dm import *
cir = Circle(Point3d(0,0,0),10)
---------------------------------
OBJECTIVE: Scale the circle by a factor of 2
YOUR CODE:
from rhino3dm import *
cir = Circle(Point3d(0,0,0),10)
cir.Radius *= 2

model = File3dm()
model.Objects.AddCircle(cir)

model.Write("export.3dm", 6)
=================================

EXAMPLE 3
=================================
CURRENT CODE:
---------------------------------
from rhino3dm import *
---------------------------------
OBJECTIVE: Generate 8 circles at origin 0,0,0. Start the radius at 1 and for each circle increase by 2.
YOUR CODE:
from rhino3dm import *

circles = []
for i in range(8):
    circles.append(Circle(Point3d(0,0,0),1+(i*2)))

model = File3dm()
for circle in circles:
    model.Objects.AddCircle(circle)

model.Write("export.3dm", 6)
=================================

The current code and objective follows. Reply with your generated code.

CURRENT CODE:
---------------------------------
$current_code
---------------------------------
OBJECTIVE: $objective
"""


class CadGPT:

    def generate(self):
        return code


if (
    __name__ == "__main__"
):
    _cadgpt = CadGPT()

    openai.api_key = os.environ["OPENAI_API_KEY"]

    def get_gpt_command(objective, current_code):
        prompt = prompt_template
        prompt = prompt.replace("$objective", objective)
        prompt = prompt.replace("$current_code", current_code)
        response = openai.Completion.create(
            model="text-curie-001", prompt=prompt, temperature=0.5, max_tokens=200, best_of=10, n=3)
        return response.choices[0].text

    objective = "Generate a sphere with radius 10 at origin (1,2,3)"
    print("Welcome to CadGPT! What is your objective?")
    i = input()
    if len(i) > 0:
        objective = i
        current_code = "from rhino3dm import *"

    print(get_gpt_command(objective, current_code))
