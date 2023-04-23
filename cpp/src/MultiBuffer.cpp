#include "MultiBuffer.h"


#include <iostream>


MultiBuffer::MultiBuffer(int width, int height, size_t count) : count(count), width(width), height(height), ptr(nullptr),
FBOHead(0), PBOHead(0), PBOTail(0)
{
	FBOs = (GLuint*)malloc(sizeof(GLuint) * count);
	RBOs = (GLuint*)malloc(sizeof(GLuint) * count);
	PBOs = (GLuint*)malloc(sizeof(GLuint) * count);
	glGenFramebuffers(count, FBOs);
	glGenRenderbuffers(count, RBOs);
	glGenBuffers(count, PBOs);
	for (int i = 0; i < count; i++)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, FBOs[i]);
		glBindRenderbuffer(GL_RENDERBUFFER, RBOs[i]);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB8, width, height);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, RBOs[i]);

		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		{
			std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
			exit(-1);
		}

		glBindBuffer(GL_PIXEL_PACK_BUFFER, PBOs[i]);
		glBufferData(GL_PIXEL_PACK_BUFFER, width * height * sizeof(uint8_t) * 3, nullptr, GL_STREAM_READ);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, FBOs[0]);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}
MultiBuffer::~MultiBuffer()
{
	glDeleteFramebuffers(count, FBOs);
	glDeleteRenderbuffers(count, RBOs);
	glDeleteBuffers(count, PBOs);
	free(FBOs);
	free(RBOs);
	free(PBOs);
}

void MultiBuffer::nextVBO()
{
	glBindBuffer(GL_PIXEL_PACK_BUFFER, PBOs[PBOHead]); PBOHead = (PBOHead + 1) % count;
	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, FBOs[FBOHead]); FBOHead = (FBOHead + 1) % count;
}
uint8_t* MultiBuffer::nextPBO()
{
	if (ptr)
	{
		glBindBuffer(GL_PIXEL_PACK_BUFFER, PBOs[mappedPBO]);
		glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
	}
	glBindBuffer(GL_PIXEL_PACK_BUFFER, PBOs[PBOTail]); mappedPBO = PBOTail; PBOTail = (PBOTail + 1) % count;
	ptr = (uint8_t*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
	return ptr;
}