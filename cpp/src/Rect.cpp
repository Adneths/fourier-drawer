#include "Rect.h"

Rect::Rect() : VAO(0), VBO(0) {}
Rect::Rect(float vertices[8])
{
	glGenBuffers(1, &VBO);
	glGenVertexArrays(1, &VAO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), vertices, GL_STREAM_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glBindVertexArray(0);
}
void Rect::draw(const GLuint shader, const glm::mat3& viewMtx, const glm::vec2& offset, const glm::vec3& color, const float linewidth, const bool fill)
{
	glUseProgram(shader);
	glLineWidth(linewidth);
	glUniformMatrix3fv(glGetUniformLocation(shader, "viewMtx"), 1, GL_FALSE, (float*)&viewMtx);
	glUniform3fv(glGetUniformLocation(shader, "DiffuseColor"), 1, (float*)&color);
	glUniform2fv(glGetUniformLocation(shader, "offset"), 1, (float*)&offset);
	glBindVertexArray(VAO);
	glDrawArrays(fill ? GL_TRIANGLE_FAN : GL_LINE_LOOP, 0, 4);
	glBindVertexArray(0);
}
Rect::~Rect()
{
	glDeleteBuffers(1, &VBO);
	glDeleteVertexArrays(1, &VAO);
}