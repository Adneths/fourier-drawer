#pragma once

#include "core.h"


class Rect {
protected:
	GLuint VBO, VAO;
public:
	Rect();
	Rect(float vertices[8]);
	void draw(const GLuint shader, const glm::mat3& viewMtx, const glm::vec2& offset, const glm::vec3& color, const float linewidth, const bool fill);
	~Rect();
};