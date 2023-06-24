#include "NpFourierSeries.h"
#include <algorithm>

void NpForuierSeries::resetTrail()
{
	std::complex<float> sum = nc::sum(vector)[0];
	last[0] = sum.real(); last[1] = sum.imag(); last[2] = 0;
	glBindBuffer(GL_ARRAY_BUFFER, pathLine->getBuffer());
	if (pathLine->isTimestamped())
		glClearBufferData(GL_ARRAY_BUFFER, GL_RGB32F, GL_RGBA, GL_FLOAT, &last);
	else
		glClearBufferData(GL_ARRAY_BUFFER, GL_RG32F, GL_RGBA, GL_FLOAT, &last);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}
NpForuierSeries::NpForuierSeries(LineStrip* vectorLine, Lines* pathLine, std::complex<float>* mags, int* freqs, size_t size, float dt, size_t cacheSize)
	: vectorLine(vectorLine), pathLine(pathLine), cacheSize(cacheSize), dt(dt), head(0)
{
	vector = nc::append({ std::complex<float>(0,0) }, nc::asarray(mags, size));
	freqsArr = nc::asarray(freqs, size);
	step = nc::append({ std::complex<float>(0,0) },
			nc::exp(std::complex<float>(0, 1) * dt * freqsArr.astype<std::complex<float>>()));

	pathCache = (float*)malloc(sizeof(float) * (cacheFloatSize = (pathLine->isTimestamped() ? 3ull : 2ull) * cacheSize * 2ull));

	glBindBuffer(GL_ARRAY_BUFFER, vectorLine->getBuffer());
	glBufferSubData(GL_ARRAY_BUFFER, 0, vectorLine->getCount() * 2ull * sizeof(float), (float*)nc::cumsum(vector).dataRelease());

	lineWidth = (pathLine->isTimestamped() ? 6ull : 4ull);
	pathBufferSize = pathLine->getCount() * lineWidth;
}
NpForuierSeries::~NpForuierSeries()
{
}

void NpForuierSeries::init(float time) {
	vector *= nc::append({ std::complex<float>(0,0) },
		nc::exp(std::complex<float>(0, 1) * time * freqsArr.astype<std::complex<float>>()));
}
float NpForuierSeries::increment(size_t count, float time)
{
	if (pathLine->isTimestamped())
	{
		for(int i = 0; i < 3; i++)
			pathCache[i] = last[i];
		for (int i = 0; i < count; i++)
		{
			vector *= step;
			std::complex<float> sum = nc::sum(vector)[0];
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
			std::complex<float> sum = nc::sum(vector)[0];
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


void NpForuierSeries::updateBuffers()
{
	glBindBuffer(GL_ARRAY_BUFFER, vectorLine->getBuffer());
	glBufferSubData(GL_ARRAY_BUFFER, 0, (vectorLine->getCount() + 1ull) * 2ull * sizeof(float), (float*)nc::cumsum(vector).dataRelease());

	for (int i = 0; i < (pathLine->isTimestamped() ? 3 : 2); i++)
		last[i] = pathCache[i + (cacheSize-1) * lineWidth + (pathLine->isTimestamped() ? 3ull : 2ull)];

	size_t len = std::min(cacheFloatSize, pathBufferSize - head);
	glBindBuffer(GL_ARRAY_BUFFER, pathLine->getBuffer());
	glBufferSubData(GL_ARRAY_BUFFER, head * sizeof(float), len * sizeof(float), pathCache);
	if (len < cacheFloatSize)
		glBufferSubData(GL_ARRAY_BUFFER, 0, (cacheFloatSize - len) * sizeof(float), pathCache + len);
	head = (head + cacheFloatSize) % (pathBufferSize);
}

void NpForuierSeries::readyBuffers() {}