#include "pt_vcglib_sampler.h"

// Torch C++Extension Lib
#include <torch/extension.h>

// VCG Libs
#include <vcg/complex/complex.h>
#include <vcg/complex/append.h>

#include <vcg/complex/algorithms/update/topology.h>
#include <vcg/complex/algorithms/update/bounding.h>
#include <vcg/complex/algorithms/update/normal.h>
#include <vcg/complex/algorithms/update/position.h>
#include <vcg/complex/algorithms/update/quality.h>
#include <vcg/complex/algorithms/stat.h>

#include <vcg/complex/algorithms/clean.h>
#include <vcg/complex/algorithms/point_sampling.h>
#include <vcg/complex/algorithms/create/resampler.h>
#include <vcg/complex/algorithms/clustering.h>
#include <vcg/simplex/face/distance.h>
#include <vcg/complex/algorithms/geodesic.h>
#include <vcg/complex/algorithms/voronoi_processing.h>

#include <vcg/space/index/grid_static_ptr.h>

#include <wrap/io_trimesh/export_off.h>
#include <wrap/io_trimesh/import_off.h>

using namespace vcg;
using namespace std;

class MyFace;
class MyEdge;
class MyVertex;

struct MyUsedTypes : public UsedTypes<Use<MyVertex>::AsVertexType, Use<MyEdge>::AsEdgeType, Use<MyFace>::AsFaceType> {};

class MyVertex : public Vertex< MyUsedTypes, vertex::Coord3f, vertex::Normal3f, vertex::Color4b, vertex::TexCoord2f,
	vertex::Qualityf, vertex::BitFlags > {};

class MyEdge : public Edge< MyUsedTypes, edge::VertexRef, edge::BitFlags, edge::EVAdj, edge::EEAdj > {};

class MyFace : public Face < MyUsedTypes, face::VertexRef, face::Normal3f, face::Color4b, face::WedgeTexCoord2f,
	face::BitFlags > {};

class MyMesh : public tri::TriMesh< std::vector<MyVertex>, std::vector<MyFace>, std::vector<MyEdge> > {};

/***** SAMPLER *****/

class BaseSampler
{
public:

	BaseSampler(MyMesh* _m) {
		m = _m;
		uvSpaceFlag = false;
		qualitySampling = false;
		perFaceNormal = false;
		//tex = 0;
	}

	MyMesh *m;
	/*QImage* tex;*/
	int texSamplingWidth;
	int texSamplingHeight;
	bool uvSpaceFlag;
	bool qualitySampling;
	bool perFaceNormal;  // default false; if true the sample normal is the face normal, otherwise it is interpolated

	void reset()
	{
		m->Clear();
	}

	void AddVert(const MyMesh::VertexType &p)
	{
		tri::Allocator<MyMesh>::AddVertices(*m, 1);
		m->vert.back().ImportData(p);
	}

	void AddFace(const MyMesh::FaceType &f, MyMesh::CoordType p)
	{
		tri::Allocator<MyMesh>::AddVertices(*m, 1);
		m->vert.back().P() = f.cP(0)*p[0] + f.cP(1)*p[1] + f.cP(2)*p[2];

		if (perFaceNormal) m->vert.back().N() = f.cN();
		else m->vert.back().N() = f.cV(0)->N()*p[0] + f.cV(1)->N()*p[1] + f.cV(2)->N()*p[2];
		if (qualitySampling)
			m->vert.back().Q() = f.cV(0)->Q()*p[0] + f.cV(1)->Q()*p[1] + f.cV(2)->Q()*p[2];
	}

	void AddTextureSample(const MyMesh::FaceType &f, const MyMesh::CoordType &p, const Point2i &tp, float edgeDist)
	{
		if (edgeDist != .0) return;

		tri::Allocator<MyMesh>::AddVertices(*m, 1);

		if (uvSpaceFlag) m->vert.back().P() = Point3<MyMesh::ScalarType>(float(tp[0]), float(tp[1]), 0);
		else m->vert.back().P() = f.cP(0)*p[0] + f.cP(1)*p[1] + f.cP(2)*p[2];

		m->vert.back().N() = f.cV(0)->N()*p[0] + f.cV(1)->N()*p[1] + f.cV(2)->N()*p[2];
		/*if (tex)
		{
			QRgb val;
			// Computing normalized texels position
			int xpos = (int)(tex->width()  * (float(tp[0]) / texSamplingWidth)) % tex->width();
			int ypos = (int)(tex->height() * (1.0 - float(tp[1]) / texSamplingHeight)) % tex->height();

			if (xpos < 0) xpos += tex->width();
			if (ypos < 0) ypos += tex->height();

			val = tex->pixel(xpos, ypos);
			m->vert.back().C() = Color4b(qRed(val), qGreen(val), qBlue(val), 255);
		}*/

	}
};

/* This sampler is used to transfer the detail of a mesh onto another one.
 * It keep internally the spatial indexing structure used to find the closest point
 */
class LocalRedetailSampler
{
	typedef GridStaticPtr<MyMesh::FaceType, MyMesh::ScalarType > MetroMeshGrid;
	typedef GridStaticPtr<MyMesh::VertexType, MyMesh::ScalarType > VertexMeshGrid;

public:

	LocalRedetailSampler() : m(0) {}

	MyMesh *m;           /// the source mesh for which we search the closest points (e.g. the mesh from which we take colors etc).
	
	int sampleNum;  // the expected number of samples. Used only for the callback
	int sampleCnt;
	MetroMeshGrid   unifGridFace;
	VertexMeshGrid   unifGridVert;
	bool useVertexSampling;

	// Parameters
	typedef tri::FaceTmark<MyMesh> MarkerFace;
	MarkerFace markerFunctor;

	bool coordFlag;
	bool colorFlag;
	bool normalFlag;
	bool qualityFlag;
	bool selectionFlag;
	bool storeDistanceAsQualityFlag;
	float dist_upper_bound;
	void init(MyMesh *_m)
	{
		coordFlag = false;
		colorFlag = false;
		qualityFlag = false;
		selectionFlag = false;
		storeDistanceAsQualityFlag = false;
		m = _m;
		tri::UpdateNormal<MyMesh>::PerFaceNormalized(*m);
		if (m->fn == 0) useVertexSampling = true;
		else useVertexSampling = false;

		if (useVertexSampling) unifGridVert.Set(m->vert.begin(), m->vert.end());
		else  unifGridFace.Set(m->face.begin(), m->face.end());
		markerFunctor.SetMesh(m);
	}

	// this function is called for each vertex of the target mesh.
	// and retrieve the closest point on the source mesh.
	void AddVert(MyMesh::VertexType &p)
	{
		assert(m);
		// the results
		Point3 < MyMesh::ScalarType> closestPt, normf, bestq, ip;
		MyMesh::ScalarType dist = dist_upper_bound;
		const MyMesh::CoordType &startPt = p.cP();
		// compute distance between startPt and the mesh S2

		MyMesh::VertexType   *nearestV = 0;
		nearestV = tri::GetClosestVertex<MyMesh, VertexMeshGrid>(*m, unifGridVert, startPt, dist_upper_bound, dist); //(PDistFunct,markerFunctor,startPt,dist_upper_bound,dist,closestPt);
		if (storeDistanceAsQualityFlag)  p.Q() = dist;
		if (dist == dist_upper_bound) return;

		if (coordFlag) p.P() = nearestV->P();
		if (colorFlag) p.C() = nearestV->C();
		if (normalFlag) p.N() = nearestV->N();
		if (qualityFlag) p.Q() = nearestV->Q();
		if (selectionFlag) if (nearestV->IsS()) p.SetS();
	}
}; // end class RedetailSampler

torch::Tensor PoissonDisk(torch::Tensor vertex, torch::Tensor faces, torch::Tensor out, unsigned int sampleNum,  float rad){

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

    vcg::tri::UpdateBounding<MyMesh>::Box(m);


   /***** SAMPLING *****/
	
	int montecarloSamples = sampleNum*10;
	int poissonDiskSamples = sampleNum;

	tri::SurfaceSampling<MyMesh, BaseSampler>::PoissonDiskParam pp;
	pp.radiusVariance = 1.0;
	bool subsampleFlag = false;
		
	float radius = tri::SurfaceSampling<MyMesh, BaseSampler>::ComputePoissonDiskRadius(m, poissonDiskSamples);
	
	MyMesh mm; // new mesh

	// generate montecarlo samples for fast lookup
	MyMesh *presampledMesh = 0;

	MyMesh MontecarloMesh;
	presampledMesh = &MontecarloMesh;

	BaseSampler sampler(presampledMesh);
	tri::SurfaceSampling<MyMesh, BaseSampler>::Montecarlo(m, sampler, montecarloSamples);
	presampledMesh->bbox = m.bbox; // we want the same bounding box
	
	BaseSampler mps(&mm);

	pp.preGenFlag = true;
	pp.geodesicDistanceFlag = true;
	pp.bestSampleChoiceFlag = true;
	pp.bestSamplePoolSize = 10;
	tri::SurfaceSampling<MyMesh, BaseSampler>::PoissonDiskPruningByNumber(mps, *presampledMesh, poissonDiskSamples, radius, pp, 0.005);


    //Update Vertex

    for(int i = 0; i<poissonDiskSamples; i++){
        out[i][0] = torch::Scalar(mm.vert[i].P()[0] );
        out[i][1] = torch::Scalar(mm.vert[i].P()[1] );
        out[i][2] = torch::Scalar(mm.vert[i].P()[2] );
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("PoissonDisk", &PoissonDisk, "Point sampling using poisson disks");
}