CMCFC_VERSION = "25060"
MINECRAFT_VERSION = "1.21.5"

default_main = """
void setup() {
  // runs on loading
}

void loop() {
  // runs every tick
}
"""

stdio_h = """
void printf(char* format);
"""

stdlib_h = """
int rand();
"""

stdlibs = {"<stdio.h>": stdio_h, "<stdlib.h>": stdlib_h}