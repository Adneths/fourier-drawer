#version 330 core

layout(location = 0) in vec2 pos;

uniform mat3 viewMtx;

flat out int ind;

void main() {
	gl_Position = vec4(viewMtx * vec3(pos, 1.0f), 1.0f);
	ind = gl_VertexID;
}