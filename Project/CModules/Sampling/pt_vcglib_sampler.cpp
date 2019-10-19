#include "pt_vcglib_sampler.h"

#include <torch/extension.h>
#include <vcg/complex/complex.h>

#include <wrap/io_trimesh/export.h>

#include <vcg/complex/algorithms/point_sampling.h>
#include <vcg/complex/algorithms/create/platonic.h>

using namespace vcg;
using namespace std;


class MyEdge;
class MyFace;
class MyVertex;
struct MyUsedTypes : public UsedTypes<	Use<MyVertex>   ::AsVertexType,
                                        Use<MyEdge>     ::AsEdgeType,
                                        Use<MyFace>     ::AsFaceType>{};

class MyVertex  : public Vertex<MyUsedTypes, vertex::Coord3f, vertex::Normal3f, vertex::BitFlags  >{};
class MyFace    : public Face< MyUsedTypes, face::FFAdj,  face::Normal3f, face::VertexRef, face::BitFlags > {};
class MyEdge    : public Edge<MyUsedTypes>{};
class MyMesh    : public tri::TriMesh< vector<MyVertex>, vector<MyFace> , vector<MyEdge>  > {};

torch::Tensor PoissonDisk(torch::Tensor vertex, torch::Tensor faces, unsigned int sampleNum,  float rad){

    unsigned int numVertex = vertex.size(0);
    unsigned int numFaces = faces.size(1);

    MyMesh m;

    tri::Allocator<MyMesh>::AddVertices(m, numVertex);
    for(unsigned int v = 0; v<numVertex; v++){
        m.vert[v].P()[0] = vertex[v][0].item().toFloat();
        m.vert[v].P()[1] = vertex[v][1].item().toFloat();
        m.vert[v].P()[2] = vertex[v][2].item().toFloat();
    }

    //Add Faces
    for(unsigned int f = 0; f<numFaces; f++){
        int p1_index = (int)faces[0][f].item().toFloat();
        int p2_index = (int)faces[1][f].item().toFloat();
        int p3_index = (int)faces[2][f].item().toFloat();
        Allocator<MyMesh>::AddFace(m, &m.vert[p1_index], &m.vert[p2_index], &m.vert[p3_index]);
    }

    tri::io::ExporterOFF<MyMesh>::Save(m,"Original.off");


    MyMesh MontecarloSurfaceMesh;
    MyMesh MontecarloEdgeMesh;
    MyMesh PoissonEdgeMesh;
    MyMesh PoissonMesh;

    /*
        Mesh recostruction and recalculation
    */
    std::vector<Point3f> sampleVec;
    tri::TrivialSampler<MyMesh> mps(sampleVec);
    tri::UpdateTopology<MyMesh>::FaceFace(m);
    tri::UpdateNormal<MyMesh>::PerFace(m);

    /*
        Calculate the edges where the angle between faces is more than 40 degree,
        filling the found edges with random points.
    */
    tri::UpdateFlags<MyMesh>::FaceEdgeSelCrease(m,math::ToRad(40.0f));
    tri::SurfaceSampling<MyMesh,tri::TrivialSampler<MyMesh> >::EdgeMontecarlo(m,mps,10000,false);
    tri::BuildMeshFromCoordVector(MontecarloEdgeMesh,sampleVec);
    tri::io::ExporterOFF<MyMesh>::Save(MontecarloEdgeMesh,"MontecarloEdgeMesh.off");

    /*
        Calculate the Crease vertex (eg: vertex more needed to define the shape)
        that we will keep as point for our sampling.
    */
    sampleVec.clear();
    tri::SurfaceSampling<MyMesh,tri::TrivialSampler<MyMesh> >::VertexCrease(m, mps);
    tri::BuildMeshFromCoordVector(PoissonEdgeMesh,sampleVec);
    tri::io::ExporterOFF<MyMesh>::Save(PoissonEdgeMesh,"VertexCreaseMesh.off");

    /*
        Cleaning of the Montecarlo generated distribution on the edges using poisson disk pruning
    */
    tri::SurfaceSampling<MyMesh,tri::TrivialSampler<MyMesh> >::PoissonDiskParam pp;
    pp.preGenMesh = &PoissonEdgeMesh;
    pp.preGenFlag=true;
    sampleVec.clear();
    tri::SurfaceSampling<MyMesh,tri::TrivialSampler<MyMesh> >::PoissonDiskPruning(mps, MontecarloEdgeMesh, rad, pp);
    tri::BuildMeshFromCoordVector(PoissonEdgeMesh,sampleVec);
    tri::io::ExporterOFF<MyMesh>::Save(PoissonEdgeMesh,"PoissonEdgeMesh.off");

    /*
        Creation of a random Montecarlo sampled cloud point on the full mesh,
        used as a base where apply the Poisson disk pruning.
    */
    sampleVec.clear();
    tri::SurfaceSampling<MyMesh,tri::TrivialSampler<MyMesh> >::Montecarlo(m,mps,50000);
    tri::BuildMeshFromCoordVector(MontecarloSurfaceMesh,sampleVec);
    tri::io::ExporterOFF<MyMesh>::Save(MontecarloSurfaceMesh,"MontecarloSurfaceMesh.off");

    /*
        Pruning of the Montecarlo generated cloud point, with Poisson disk pruning,
        using the EdgeSampling of before as a base.
    */
    pp.preGenMesh = &PoissonEdgeMesh;
    pp.preGenFlag=true;
    sampleVec.clear();
    tri::SurfaceSampling<MyMesh, tri::TrivialSampler<MyMesh> >::PoissonDiskPruning(mps, MontecarloSurfaceMesh, rad, pp);
    tri::BuildMeshFromCoordVector(PoissonMesh,sampleVec);
    tri::io::ExporterOFF<MyMesh>::Save(PoissonMesh,"PoissonMesh.off");

    //Update Vertex
    /*
    for(unsigned int i = 0; i<sampleNum; i++){
        vertex[i][0] = torch::Scalar(m.vert[i].P()[0] );
        vertex[i][1] = torch::Scalar(m.vert[i].P()[1] );
        vertex[i][2] = torch::Scalar(m.vert[i].P()[2] );
    }*/

    return vertex;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("PoissonDisk", &PoissonDisk, "Point sampling using poisson disks");
}