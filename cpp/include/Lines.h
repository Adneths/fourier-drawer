#pragma once

#include "core.h"

using namespace math;

template <typename T>
class Lines {
protected:
	bool timestamped;
	GLuint VBO, VAO, fillShader;
	size_t count;
public:
	/**
	 * @param vertex the vertex value to copy into every vertex
	 * @param count the number of lines
	 * @param the color of the line strip
	 */
	Lines(vec2<T> vertex, uint32_t count, bool timestamped, GLuint fillShader);
	/**
	 * @param vertices the vertex values to be copied
	 * @param count the number of lines
	 * @param the color of the line strip
	 */
	//Lines(float* vertices, uint32_t count, bool timestamped);
	void draw(const GLuint shader, const mat3<T> &viewMtx, const vec2<T>& offset, const fvec3 &color, const float linewidth);
	~Lines();

	void fill(vec3<T> val);
	void fill(T val[3]);

	GLuint getBuffer();
	/**
	 * @return the number of lines
	 */
	uint32_t getCount();
	bool isTimestamped() {
		return timestamped;
	}
};