#pragma once

#include "core.h"

using namespace math;

template <typename T>
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
	LineStrip(vec2<T> vertex, uint32_t count);
	/**
	 * @param vertices the vertex values to be copied
	 * @param count the number of line segments
	 * @param the color of the line strip
	 */
	LineStrip(T* vertices, uint32_t length);
	void draw(const GLuint shader, const mat3<T> &viewMtx, const vec2<T> offset, const fvec3& color, const float linewidth);
	~LineStrip();

	GLuint getBuffer();
	/**
	 * @return the number of line segments
	 */
	uint32_t getCount();
};