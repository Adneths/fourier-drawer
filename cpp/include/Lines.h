#pragma once

#include "core.h"


class Lines {
protected:
	bool timestamped;
	GLuint VBO, VAO;
	size_t count;
public:
	/**
	 * @param vertex the vertex value to copy into every vertex
	 * @param count the number of lines
	 * @param the color of the line strip
	 */
	Lines(glm::vec2 vertex, uint32_t count, bool timestamped);
	/**
	 * @param vertices the vertex values to be copied
	 * @param count the number of lines
	 * @param the color of the line strip
	 */
	Lines(float* vertices, uint32_t count, bool timestamped);
	virtual void draw(GLuint shader, glm::mat3 viewMtx);
	virtual ~Lines();

	virtual GLuint getBuffer();
	/**
	 * @return the number of lines
	 */
	uint32_t getCount();
	bool isTimestamped() {
		return timestamped;
	}
};