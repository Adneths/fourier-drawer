#version 330 core

layout(location = 0) in vec2 pos;
layout(location = 1) in float tin;

uniform mat3 viewMtx;
uniform vec2 offset;

out float t;

void main() {
	gl_Position = vec4(viewMtx * vec3(pos, 1.0f) + vec3(offset, 0.0f), 1.0f);
	t = tin;
}