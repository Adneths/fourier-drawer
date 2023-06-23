#include "LineStrip.h"


LineStrip::LineStrip(glm::vec2 vertex, uint32_t count, glm::vec3 color)
{
	this->count = count;
	this->color = color;

	glGenBuffers(1, &VBO);
	glGenVertexArrays(1, &VAO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, 2 * (count+1) * sizeof(float), nullptr, GL_STREAM_DRAW);
	glClearBufferData(GL_ARRAY_BUFFER, GL_RG32F, GL_RGBA, GL_FLOAT, &vertex);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
}
LineStrip::LineStrip(float* vertices, uint32_t count, glm::vec3 color)
{
	this->count = count;
	this->color = color;

	glGenBuffers(1, &VBO);
	glGenVertexArrays(1, &VAO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, 2 * (count+1) * sizeof(float), vertices, GL_STREAM_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
}
void LineStrip::draw(GLuint shader, glm::mat3 viewMtx, float time)
{
	glUseProgram(shader);
	glUniformMatrix3fv(glGetUniformLocation(shader, "viewMtx"), 1, GL_FALSE, (float*)&viewMtx);
	glUniform3fv(glGetUniformLocation(shader, "DiffuseColor"), 1, (float*)&color);
	glBindVertexArray(VAO);
	glDrawArrays(GL_LINE_STRIP, 0, this->count+1);
}
LineStrip::~LineStrip()
{
	glDeleteBuffers(1, &VBO);
	glDeleteVertexArrays(1, &VAO);
}
GLuint LineStrip::getBuffer()
{
	return VBO;
}
uint32_t LineStrip::getCount()
{
	return count;
}