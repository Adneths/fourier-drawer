#include "Lines.h"


Lines::Lines(glm::vec2 vertex, uint32_t count, bool timestamped) : timestamped(timestamped)
{
	this->count = count;

	glGenBuffers(1, &VBO);
	glGenVertexArrays(1, &VAO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, (timestamped ? 6ull : 4ull) * count * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
	glClearBufferData(GL_ARRAY_BUFFER, GL_RG32F, GL_RGBA, GL_FLOAT, &vertex);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, (timestamped ? 3ull : 2ull) * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	if (timestamped)
	{
		glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3ull * sizeof(float), (void*)(2 * sizeof(float)));
		glEnableVertexAttribArray(1);
	}
	glBindVertexArray(0);
}
Lines::Lines(float* vertices, uint32_t count, bool timestamped) : timestamped(timestamped)
{
	this->count = count;

	glGenBuffers(1, &VBO);
	glGenVertexArrays(1, &VAO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	if(timestamped)
	{
		float* data = (float*)malloc(sizeof(float) * 6ull * count);
		for (int i = 0; i < count; i++)
		{
			data[i * 3 + 0] = vertices[i * 2 + 0];
			data[i * 3 + 1] = vertices[i * 2 + 1];
			data[i * 3 + 2] = 0;
		}
		glBufferData(GL_ARRAY_BUFFER, 6ull * count * sizeof(float), data, GL_DYNAMIC_DRAW);
		free(data);
	}
	else
	{
		glBufferData(GL_ARRAY_BUFFER, 4ull * count * sizeof(float), vertices, GL_DYNAMIC_DRAW);
	}

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, (timestamped ? 3ull : 2ull) * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	if (timestamped)
	{
		glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3ull * sizeof(float), (void*)(2 * sizeof(float)));
		glEnableVertexAttribArray(1);
	}
	glBindVertexArray(0);
}
void Lines::draw(GLuint shader, glm::mat3 viewMtx)
{
	glUseProgram(shader);
	glUniformMatrix3fv(glGetUniformLocation(shader, "viewMtx"), 1, GL_FALSE, (float*)&viewMtx);
	glBindVertexArray(VAO);
	glDrawArrays(GL_LINES, 0, this->count * 2);
	glBindVertexArray(0);
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
uint32_t Lines::getCount()
{
	return count;
}