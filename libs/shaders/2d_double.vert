#version 410 core

layout(location = 0) in dvec2 pos;
layout(location = 1) in double tin;

uniform dmat3 viewMtx;
uniform dvec2 offset;

flat out double t;

void main() {
	gl_Position = vec4(viewMtx * dvec3(pos, 1.0) + dvec3(offset, 0.0), 1.0);
	t = tin;
}