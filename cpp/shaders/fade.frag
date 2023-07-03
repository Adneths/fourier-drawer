#version 330 core

in float t;

uniform vec3 DiffuseColor;
uniform float time;
uniform float trailLength;

out vec4 fragColor;

void main()
{
	fragColor = vec4(DiffuseColor, 1.0f-(time-t)/trailLength);
}