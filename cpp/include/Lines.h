#pragma once

#include "core.h"

class Lines {
protected:
	GLuint VBO, VAO;
	glm::vec3 color;
	size_t count;
public:
	/**
	 * @param vertex the vertex value to copy into every vertex
	 * @param count the number of lines
	 * @param the color of the line strip
	 */
	Lines(glm::vec2 vertex, uint32_t count, glm::vec3 color);
	/**
	 * @param vertices the vertex values to be copied
	 * @param count the number of lines
	 * @param the color of the line strip
	 */
	Lines(float* vertices, uint32_t count, glm::vec3 color);
	virtual void draw(GLuint shader, glm::mat3 viewMtx);
	virtual ~Lines();

	virtual GLuint getBuffer();
	virtual void finish();
	/**
	 * @return the number of lines
	 */
	uint32_t getCount();
};