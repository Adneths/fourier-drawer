#define PY_SSIZE_T_CLEAN
#include "python3.9/Python.h"
#include "float.h"
#include <tuple>
#include <complex>

typedef struct P {
	std::complex<double>* z;
	int id;
	int ind;
	P(std::complex<double>* z, int id, int ind) : z(z), id(id), ind(ind)
    { }
} point;

int compareX(const void* a, const void* b)
{
	double d = (**(point**)a).z->real() - (**(point**)b).z->real();
	if (d < 0) return -1;
	if (d > 0) return 1;
	return 0;
}
int compareY(const void* a, const void* b)
{
	double d = (**(point**)a).z->imag() - (**(point**)b).z->imag();
	if (d < 0) return -1;
	if (d > 0) return 1;
	return 0;
}

std::tuple<point*,point*,double> findClosestIn(point** X, point** Y, int len, int total, point** reserve)
{
	//Base case
	if(len < 60)
	{
		for(int k = 1; k < len; k++)
			if(X[0]->id != X[k]->id)
			{
				point *n, *m; n = m = nullptr;
				double minD = DBL_MAX;
				for(int i = 0; i < len; i++)
					for(int j = i+1; j < len; j++)
					{
						if(X[i]->id != X[j]->id)
						{
							double d = std::abs(*(X[i]->z) - *(X[j]->z));
							if(d < minD)
							{
								minD = d; n = X[i]; m = X[j];
							}
						}
					}
				return std::make_tuple(n,m,minD);
			}
		return std::make_tuple(nullptr,nullptr,DBL_MAX);
	}
	
	//Divide and Conquer
	int mid = len/2;
	double midX = X[mid]->z->real();
	
	std::tuple<point*,point*,double> left = findClosestIn(X,Y,mid, total, reserve);
	std::tuple<point*,point*,double> right = findClosestIn(X+mid,Y,len-mid, total, reserve);
	point *n, *m;
	double minD;
	if(std::get<2>(left) > std::get<2>(right))
	{
		n = std::get<0>(right); m = std::get<1>(right); minD = std::get<2>(right);
	}
	else
	{
		n = std::get<0>(left); m = std::get<1>(left); minD = std::get<2>(left);
	}
	
	//Extract strip
	int count = 0;
	for(int i = 0; i < total; i++)
		if(abs(Y[i]->z->real() - midX) < minD)
			reserve[count++] = Y[i];
	
	for(int k = 1; k < count; k++)
		if(reserve[0]->id != reserve[k]->id)
		{
			for(int i = 0; i < count; i++)
				for(int j = i+1; j < count && (reserve[j]->z->imag() - reserve[i]->z->imag()) < minD; j++)
				{
					if(reserve[i]->id != reserve[j]->id)
					{
						double d = std::abs(*(reserve[i]->z) - *(reserve[j]->z));
						if(d < minD)
						{
							n = reserve[i]; m = reserve[j]; minD = d;
						}
					}
				}
			break;
		}
	return std::make_tuple(n,m,minD);
}

extern "C" {
    PyObject* findClosest(int lenA, std::complex<double>* A, int lenB, std::complex<double>* B)
    {
		point** cABx = (point**)PyMem_RawMalloc((lenA+lenB) * sizeof(point*));
		point** cABy = (point**)PyMem_RawMalloc((lenA+lenB) * sizeof(point*));
		point** reserve = (point**)PyMem_RawMalloc((lenA+lenB) * sizeof(point*));
		for(int i = 0; i < lenA; i++)
		{
			point* p = new point(&A[i], 0, i);
			cABx[i] = p;
			cABy[i] = p;
		}
		for(int i = 0; i < lenB; i++)
		{
			point* p = new point(&B[i], 1, i);
			cABx[lenA+i] = p;
			cABy[lenA+i] = p;
		}
		qsort(cABx, lenA+lenB, sizeof(point*), compareX);
		qsort(cABy, lenA+lenB, sizeof(point*), compareY);
		
		std::tuple<point*,point*,double> pair = findClosestIn(cABx, cABy, lenA+lenB, lenA+lenB, reserve);
		PyObject* data;
		if(std::get<0>(pair)->id == 0)
			data = Py_BuildValue("(iid)",std::get<0>(pair)->ind,std::get<1>(pair)->ind,std::get<2>(pair));
		else
			data = Py_BuildValue("(iid)",std::get<1>(pair)->ind,std::get<0>(pair)->ind,std::get<2>(pair));
		
		for(int i = 0; i < lenA+lenB; i++)
			delete cABx[i];
		PyMem_RawFree(cABx);
		PyMem_RawFree(cABy);
		PyMem_RawFree(reserve);
		return data;
    }
}