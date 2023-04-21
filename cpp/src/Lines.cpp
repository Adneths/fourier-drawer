#include "Lines.h"


Lines::Lines(glm::vec2 vertex, uint32_t count, glm::vec3 color)
{
	this->count = count;
	this->color = color;

	glGenBuffers(1, &VBO);
	glGenVertexArrays(1, &VAO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, 4 * count * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
	glClearBufferData(GL_ARRAY_BUFFER, GL_RG32F, GL_RGBA, GL_FLOAT, &vertex);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
}
Lines::Lines(float* vertices, uint32_t count, glm::vec3 color)
{
	this->count = count;
	this->color = color;

	glGenBuffers(1, &VBO);
	glGenVertexArrays(1, &VAO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, 4 * count * sizeof(float), vertices, GL_DYNAMIC_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
}
void Lines::draw(GLuint shader, glm::mat3 viewMtx)
{
	glUseProgram(shader);
	glUniformMatrix3fv(glGetUniformLocation(shader, "viewMtx"), 1, GL_FALSE, (float*)&viewMtx);
	glUniform3fv(glGetUniformLocation(shader, "DiffuseColor"), 1, (float*)&color);
	glBindVertexArray(VAO);
	glDrawArrays(GL_LINES, 0, this->count * 2);
}
Lines::~Lines()
{
	glDeleteBuffers(1, &VBO);
	glDeleteVertexArrays(1, &VAO);
}
GLuint Lines::getBuffer()
{
	return VBO;
}
void Lines::finish()
{

}
uint32_t Lines::getCount()
{
	return count;
}