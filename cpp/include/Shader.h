#pragma once

#include "core.h"

#include <stdio.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

GLuint LoadShaders(const char* vertex_shader, const char* fragment_shade, bool debug);
