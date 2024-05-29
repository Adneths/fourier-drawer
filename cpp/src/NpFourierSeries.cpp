#include "NpFourierSeries.h"
#include <algorithm>
template class NpFourierSeries<float>;
template class NpFourierSeries<double>;

template <typename T>
void NpFourierSeries<T>::resetTrail(vec2<T>* vecHeadPtr)
{
	std::complex<T> sum = nc::sum(vector)[0];
	last[0] = sum.real(); last[1] = sum.imag(); last[2] = 0;
	pathLine->fill(last);
	
	if (vecHeadPtr != nullptr)
		*vecHeadPtr = vec2<T>(last[0], last[1]);
}
template <typename T>
NpFourierSeries<T>::NpFourierSeries(LineStrip<T>* vectorLine, Lines<T>* pathLine, std::complex<T>* mags, int* freqs, size_t size, T dt, size_t cacheSize)
	: vectorLine(vectorLine), pathLine(pathLine), cacheSize(cacheSize), dt(dt), head(0), last{0.0,0.0,0.0}
{
	vector = nc::append({ std::complex<T>(0,0) }, nc::asarray(mags, size));
	freqsArr = nc::asarray(freqs, size);
	step = nc::append({ std::complex<T>(0,0) },
			nc::exp(std::complex<T>(0, 1) * dt * freqsArr.astype<std::complex<T>>()));

	pathCache = (T*)malloc(sizeof(T) * (cacheFloatSize = (pathLine->isTimestamped() ? 3ull : 2ull) * cacheSize * 2ull));

	glBindBuffer(GL_ARRAY_BUFFER, vectorLine->getBuffer());
	glBufferSubData(GL_ARRAY_BUFFER, 0, (vectorLine->getCount() + 1) * 2ull * sizeof(T), (T*)nc::cumsum(vector).dataRelease());

	lineWidth = (pathLine->isTimestamped() ? 6ull : 4ull);
	pathBufferSize = pathLine->getCount() * lineWidth;
}
template <typename T>
NpFourierSeries<T>::~NpFourierSeries()
{
}

template <typename T>
void NpFourierSeries<T>::init(T time) {
	vector *= nc::append({ std::complex<T>(0,0) },
		nc::exp(std::complex<T>(0, 1) * time * freqsArr.astype<std::complex<T>>()));
}
template <typename T>
T NpFourierSeries<T>::increment(size_t count, T time)
{
	if (pathLine->isTimestamped())
	{
		for(int i = 0; i < 3; i++)
			pathCache[i] = last[i];
		for (int i = 0; i < count; i++)
		{
			vector *= step;
			std::complex<T> sum = nc::sum(vector)[0];
			pathCache[i * 6 + 0 + 3] = sum.real();
			pathCache[i * 6 + 1 + 3] = sum.imag();
			pathCache[i * 6 + 2 + 3] = time + dt * (i + 1);
			if (i != count - 1)
			{
				pathCache[i * 6 + 3 + 3] = sum.real();
				pathCache[i * 6 + 4 + 3] = sum.imag();
				pathCache[i * 6 + 5 + 3] = time + dt * (i + 1);
			}
		}
	}
	else
	{
		for (int i = 0; i < 2; i++)
			pathCache[i] = last[i];
		for (int i = 0; i < count; i++)
		{
			vector *= step;
			std::complex<T> sum = nc::sum(vector)[0];
			pathCache[i * 4 + 0 + 2] = sum.real();
			pathCache[i * 4 + 1 + 2] = sum.imag();
			if (i != count - 1)
			{
				pathCache[i * 4 + 2 + 2] = sum.real();
				pathCache[i * 4 + 3 + 2] = sum.imag();
			}
		}
	}

	return count * dt;
}


template <typename T>
void NpFourierSeries<T>::updateBuffers(vec2<T>* vecHeadPtr)
{
	glBindBuffer(GL_ARRAY_BUFFER, vectorLine->getBuffer());
	glBufferSubData(GL_ARRAY_BUFFER, 0, (vectorLine->getCount() + 1ull) * 2ull * sizeof(T), (T*)nc::cumsum(vector).dataRelease());

	for (int i = 0; i < (pathLine->isTimestamped() ? 3 : 2); i++)
		last[i] = pathCache[i + (cacheSize-1) * lineWidth + (pathLine->isTimestamped() ? 3ull : 2ull)];

	size_t len = std::min(cacheFloatSize, pathBufferSize - head);
	glBindBuffer(GL_ARRAY_BUFFER, pathLine->getBuffer());
	glBufferSubData(GL_ARRAY_BUFFER, head * sizeof(T), len * sizeof(T), pathCache);
	if (len < cacheFloatSize)
		glBufferSubData(GL_ARRAY_BUFFER, 0, (cacheFloatSize - len) * sizeof(T), pathCache + len);
	head = (head + cacheFloatSize) % (pathBufferSize);

	if (vecHeadPtr != nullptr)
		*vecHeadPtr = vec2<T>(last[0], last[1]);
}

template <typename T>
void NpFourierSeries<T>::readyBuffers() {
}