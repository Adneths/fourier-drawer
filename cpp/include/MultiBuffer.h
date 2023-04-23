#pragma once

#include "core.h"

class MultiBuffer {
private:
	size_t count;
	int width, height;
	GLuint* FBOs;
	GLuint* RBOs;
	GLuint* PBOs;

	uint8_t* ptr;
	int FBOHead, PBOHead, PBOTail, mappedPBO;
public:
	MultiBuffer(int width, int height, size_t count);
	~MultiBuffer();

	void nextVBO();
	GLubyte* nextPBO();
};