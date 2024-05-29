#version 400 core

flat in double t;

uniform vec3 DiffuseColor;
uniform double time;
uniform double pathLength;

out vec4 fragColor;

void main()
{
	fragColor = vec4(DiffuseColor, 1.0f-(time-t)/pathLength);
}