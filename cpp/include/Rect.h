#pragma once

#include "core.h"

using namespace math;

template <typename T>
class Rect {
protected:
	GLuint VBO, VAO;
public:
	Rect();
	Rect(T vertices[8]);
	void draw(const GLuint shader, const mat3<T>& viewMtx, const vec2<T>& offset, const fvec3& color, const float linewidth, const bool fill);
	~Rect();
};