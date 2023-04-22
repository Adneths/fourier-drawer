#pragma once

#include "core.h"

class LineStrip {
protected:
	GLuint VBO, VAO;
	glm::vec3 color;
	size_t count;
public:
	/**
	 * @param vertex the vertex value to copy into every vertex
	 * @param count the number of line segments
	 * @param the color of the line strip
	 */
	LineStrip(glm::vec2 vertex, uint32_t count, glm::vec3 color);
	/**
	 * @param vertices the vertex values to be copied
	 * @param count the number of line segments
	 * @param the color of the line strip
	 */
	LineStrip(float* vertices, uint32_t length, glm::vec3 color);
	virtual void draw(GLuint shader, glm::mat3 viewMtx, float time = 0);
	virtual ~LineStrip();

	virtual GLuint getBuffer();
	virtual void finish();
	/**
	 * @return the number of line segments
	 */
	uint32_t getCount();
};