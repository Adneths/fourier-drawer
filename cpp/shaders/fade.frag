#version 330 core

flat in int ind;

uniform vec3 DiffuseColor;
uniform int total;

out vec4 fragColor;

void main()
{
	fragColor = vec4(DiffuseColor, float(ind)/total);
}