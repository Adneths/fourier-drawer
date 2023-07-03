#pragma once

#include "core.h"

class LineStrip {
protected:
	GLuint VBO, VAO;
	size_t count;
public:
	/**
	 * @param vertex the vertex value to copy into every vertex
	 * @param count the number of line segments
	 * @param the color of the line strip
	 */
	LineStrip(glm::vec2 vertex, uint32_t count);
	/**
	 * @param vertices the vertex values to be copied
	 * @param count the number of line segments
	 * @param the color of the line strip
	 */
	LineStrip(float* vertices, uint32_t length);
	virtual void draw(GLuint shader, glm::mat3 viewMtx);
	virtual ~LineStrip();

	virtual GLuint getBuffer();
	/**
	 * @return the number of line segments
	 */
	uint32_t getCount();
};