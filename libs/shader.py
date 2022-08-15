from ctypes import *
from OpenGL.GL import *
import OpenGL.GL.shaders

VERTEX_SHADER = '''
#version 430

layout(location = 0) in vec2 pos;
layout(location = 2) uniform mat2 scale;

flat out int ind;

void main() {
	gl_Position = vec4(scale * pos, 0.0, 1.0);
	ind = gl_VertexID;
}
'''

FRAGMENT_SHADER = '''
#version 430
out vec4 FragColor;

layout(location = 0) uniform vec3 drawColor;

void main()
{
	FragColor = vec4(drawColor, 1.0);
}
'''

FRAGMENT_SHADER_FADE = '''
#version 430
out vec4 FragColor;

flat in int ind;
layout(location = 0) uniform vec3 drawColor;
layout(location = 1) uniform int total;

void main()
{
	FragColor = vec4(drawColor, float(ind)/total);
}
'''

FRAGMENT_SHADER_FADE_DITHER = '''
#version 430
out vec4 FragColor;

flat in int ind;
layout(location = 0) uniform vec3 drawColor;
layout(location = 1) uniform int total;
layout(location = 3) uniform mat4 map;
layout(location = 4) uniform int colors;

void main()
{
	double m = map[int(mod(gl_FragCoord.x,4))][int(mod(gl_FragCoord.y,4))];
	FragColor = vec4(drawColor, round((float(ind)/total + m/colors)*(colors-1))/(colors-1));
}
'''
	
def getShaders(isGif, pathLength, pathFade, dims, bpp):
	shader_vec = OpenGL.GL.shaders.compileProgram(
		OpenGL.GL.shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
		OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
	)
	M = (c_float * 4)(2/dims[0],0, 0,2/dims[1])
	glUseProgram(shader_vec)
	glUniformMatrix2fv(2, 1, GL_FALSE, pointer(M))
	
	if pathFade:
		if isGif:
			shader_path = OpenGL.GL.shaders.compileProgram(
				OpenGL.GL.shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
				OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER_FADE_DITHER, GL_FRAGMENT_SHADER),
			)
			glUseProgram(shader_path)
			threshold = (c_float * 16)(0,1/2,1/8,5/8,3/4,1/4,7/8,3/8,3/16,11/16,1/16,9/16,15/16,7/16,13/16,5/16)
			glUniformMatrix4fv(3, 1, GL_FALSE, pointer(threshold))
			glUniform1i(4, 2**bpp-2)
		else:
			shader_path = OpenGL.GL.shaders.compileProgram(
				OpenGL.GL.shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
				OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER_FADE, GL_FRAGMENT_SHADER),
			)
			glUseProgram(shader_path)
		glUniformMatrix2fv(2, 1, GL_FALSE, pointer(M))
		glUniform1i(1, pathLength)
	else:
		shader_path = shader_vec
	
	return shader_vec, shader_path