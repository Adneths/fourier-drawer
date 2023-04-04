#pragma once

#include "LineStrip.h"
#include "core.h"

class CyclicalLineStrip : LineStrip {
public:
	size_t head;
	CyclicalLineStrip(glm::vec2 vertex, size_t count, glm::vec3 color);
	void draw(GLuint shader, glm::mat3 viewMtx);
	~CyclicalLineStrip();

	GLuint getBuffer();
	void finish();
};