#include "NpFourierSeries.h"


NpForuierSeries::NpForuierSeries(LineStrip* vectorLine, CyclicalLineStrip* pathLine, std::complex<float>* series, size_t size, float dt, size_t cacheSize)
	: vectorLine(vectorLine), pathLine(pathLine), dt(dt), cacheSize(cacheSize)
{
	nc::NdArray<std::complex<float>> mags((std::complex<float>*)series, size);
	nc::NdArray<int> freqs = nc::append(nc::arange(0, (int)(size / 2)), nc::arange(-(int)((size + 1) / 2), 0));
	nc::NdArray<nc::uint32> inds = nc::argsort(-nc::abs(mags));
	vector = nc::append({ std::complex<float>(0,0) }, mags[inds]);
	freqs = freqs[inds];
	step = nc::append({ std::complex<float>(0,0) },
			nc::exp(std::complex<float>(0, 1) * dt * freqs.astype<std::complex<float>>()));

	pathCache = nc::empty<std::complex<float>>(cacheSize, 1);


	glBindBuffer(GL_ARRAY_BUFFER, vectorLine->getBuffer());
	glBufferSubData(GL_ARRAY_BUFFER, 0, vectorLine->getCount() * 2 * sizeof(float), (float*)nc::cumsum(vector).dataRelease());
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
void NpForuierSeries::updateBuffers()
{
	glBindBuffer(GL_ARRAY_BUFFER, vectorLine->getBuffer());
	glBufferSubData(GL_ARRAY_BUFFER, 0, vectorLine->getCount() * 2 * sizeof(float), (float*)nc::cumsum(vector).dataRelease());

	vectorLine->finish();
	pathLine->finish();
}