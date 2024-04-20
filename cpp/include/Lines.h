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
	//Lines(float* vertices, uint32_t count, bool timestamped);
	void draw(const GLuint shader, const glm::mat3 &viewMtx, const glm::vec2& offset, const glm::vec3 &color, const float linewidth);
	~Lines();

	GLuint getBuffer();
	/**
	 * @return the number of lines
	 */
	uint32_t getCount();
	bool isTimestamped() {
		return timestamped;
	}
};