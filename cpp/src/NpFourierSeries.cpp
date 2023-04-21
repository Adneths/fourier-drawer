#include "NpFourierSeries.h"
#include <algorithm>


NpForuierSeries::NpForuierSeries(LineStrip* vectorLine, Lines* pathLine, std::complex<float>* series, size_t size, float dt, size_t cacheSize)
	: vectorLine(vectorLine), pathLine(pathLine), dt(dt), cacheSize(cacheSize), head(0)
{
	nc::NdArray<std::complex<float>> mags((std::complex<float>*)series, size);
	nc::NdArray<int> freqs = nc::append(nc::arange(0, (int)(size / 2)), nc::arange(-(int)((size + 1) / 2), 0));
	nc::NdArray<nc::uint32> inds = nc::argsort(-nc::abs(mags));
	vector = nc::append({ std::complex<float>(0,0) }, mags[inds]);
	freqs = freqs[inds];
	step = nc::append({ std::complex<float>(0,0) },
			nc::exp(std::complex<float>(0, 1) * dt * freqs.astype<std::complex<float>>()));

	pathCache = nc::empty<std::complex<float>>(cacheSize, 1);

	last = nc::sum(vector)[0];
	glBindBuffer(GL_ARRAY_BUFFER, vectorLine->getBuffer());
	glBufferSubData(GL_ARRAY_BUFFER, 0, vectorLine->getCount() * 2ull * sizeof(float), (float*)nc::cumsum(vector).dataRelease());
	glBindBuffer(GL_ARRAY_BUFFER, pathLine->getBuffer());
	glClearBufferData(GL_ARRAY_BUFFER, GL_RG32F, GL_RGBA, GL_FLOAT, &last);
}
NpForuierSeries::~NpForuierSeries()
{
}

float NpForuierSeries::increment(size_t count)
{
	for (int i = 0; i < count; i++)
	{
		vector *= step;
		pathCache[i] = nc::sum(vector)[0];
	}

	return count * dt;
}

int i = 0;
void NpForuierSeries::updateBuffers()
{
	glBindBuffer(GL_ARRAY_BUFFER, vectorLine->getBuffer());
	glBufferSubData(GL_ARRAY_BUFFER, 0, (vectorLine->getCount() + 1) * 2ull * sizeof(float), (float*)nc::cumsum(vector).dataRelease());
	vectorLine->finish();

	nc::NdArray<std::complex<float>> init = nc::empty<std::complex<float>>(1, 1); init(0, 0) = last; last = pathCache[cacheSize-1];
	size_t cacheFloatSize = cacheSize * 4ull; size_t pathBufferSize = pathLine->getCount() * 4ull;
	//Beacuse NumCpp repeat does not work correctly
	nc::NdArray<std::complex<float>> lines = nc::reshape(nc::transpose(nc::repeat(nc::append(init, pathCache), nc::Shape(2, 1))), nc::Shape(1, -1));
	lines = lines(0, nc::Slice(1, -1));
	float* data = (float*)lines.dataRelease();

	size_t len = std::min(cacheFloatSize, pathBufferSize - head);
	glBindBuffer(GL_ARRAY_BUFFER, pathLine->getBuffer());
	glBufferSubData(GL_ARRAY_BUFFER, head * sizeof(float), len * sizeof(float), data);
	if (len < cacheFloatSize)
		glBufferSubData(GL_ARRAY_BUFFER, 0, (cacheFloatSize - len) * sizeof(float), data+len);
	head = (head + cacheFloatSize) % (pathLine->getCount() * 4ull);
	
	pathLine->finish();
}