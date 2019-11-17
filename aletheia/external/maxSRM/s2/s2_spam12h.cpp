#include "../submodel.h"
#include "../mat2D.h"
#include "../config.cpp"

class s2_spam12h: public Submodel
{
public:
	s2_spam12h(float q, Config *config) : Submodel(q) 
	{
		this->modelName =  "s2_spam12h";
		this->mergeInto = "s2_spam12hv";
		this->minmax = false;
		this->coocDirs.push_back("hor");
		this->coocDirs.push_back("ver");

		Initialize(config);
	}

	~s2_spam12h()
	{
	}

	virtual void ComputeFea(std::vector<mat2D<int> *> QResVect)
	{
		std::vector<std::vector<mat2D<int> *> > OpVect;

		// [0] - Horizontal, [1] - Vertical, [2] - Diagonal, [3] - Minor Diagonal

		// Horizontal
		std::vector<mat2D<int> *> H = std::vector<mat2D<int> *>();
		H.push_back(QResVect[0]);
		OpVect.push_back(H);

		// Vertical
		std::vector<mat2D<int> *> V = std::vector<mat2D<int> *>();
		V.push_back(QResVect[1]);
		OpVect.push_back(V);

		this->AddFea(OpVect);
	}
};
