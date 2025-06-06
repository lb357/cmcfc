# CMCFC
CMCFC is compiler/translator from C to MCFunction (Minecraft Datapacks)
> [!WARNING]
> Early stage of development.

## Usage
CMCFC creates minecraft datapack that can be assembled from C code.
See `python main.py --help`

## Features
List of features:
- [x] CMCFC
  - [x] preprocessor
  - [x] compiler
  - [x] linker (Datapack manager)
- [ ] Types
  - [x] simple datatypes: int, float, dummy
  - [x] simple pointers
  - [x] arrays
  - [x] functions
  - [x] native minecraft functions (by `asm` keyword)
  - [ ] complex datatypes: uint8_t, etc.
  - [ ] complex pointers (2+ dimensional ArrayRef, ptr to ptr, etc.)
  - [ ] structs
  - [ ] enum
  - [ ] typedef
  - [ ] union
- [ ] Operations
  - [x] mathematical and logical operations
  - [x] pointer operations
  - [ ] bitwise operations
- [ ] Directives
  - [x] #define
  - [x] #include
  - [ ] #pragma
  - [ ] #ifdef
  - [ ] etc.
- [ ] Standard Library (STD)
  - [x] simple standard funcitons (`printf`, `rand`, etc.)
  - [ ] complex standard functions (`cos`, `sqrt`, etc.)
  - [ ] file functions
  - [ ] minecraft API functions

## Standard Library (STD)
Includes:
1. <stdio.h>:
```c
void printf(char* format);
```
2. <stdlib.h>:
```c
int rand();
```

## Example code
Random array sorting script.

File `main.c` (datapack/src):
```c
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 16

int x[ARRAY_SIZE];

void display_array(){
    for (int i = 0; i < ARRAY_SIZE; i++){
        int y = x[i];
        printf("Index: %s / Value: %s", i, y);
    }
}

void setup() {
    for (int i = 0; i < ARRAY_SIZE; i++){
        int y = rand();
        x[i] = y;
    }
    printf("RANDOM ARRAY:");
    display_array();

    int flag = 0;
    int counter = 0;
    do {
        flag = 0;
        for (int i = 1; i < ARRAY_SIZE; i++){
            int left = x[i-1];
            int right = x[i];
            if (left > right){
                int temp = left;
                left = right;
                right = temp;
                x[i-1] = left;
                x[i] = right;
                flag = 1;
            }
        }
        counter++;
    } while (flag);

    printf("SORTED ARRAY (%s iterations):", counter);
    display_array();
}
void loop() {}
```
Input (Minecraft chat):
```mcfunction
/reload
```
Output (Minecraft chat):
```text
[System] [CHAT] RANDOM ARRAY:
[System] [CHAT] Index: 0 / Value: 49401
[System] [CHAT] Index: 1 / Value: 7529
[System] [CHAT] Index: 2 / Value: 112477
[System] [CHAT] Index: 3 / Value: 112280
[System] [CHAT] Index: 4 / Value: 168304
[System] [CHAT] Index: 5 / Value: 16916
[System] [CHAT] Index: 6 / Value: 122125
[System] [CHAT] Index: 7 / Value: 125808
[System] [CHAT] Index: 8 / Value: 93711
[System] [CHAT] Index: 9 / Value: 195541
[System] [CHAT] Index: 10 / Value: 1487
[System] [CHAT] Index: 11 / Value: 177818
[System] [CHAT] Index: 12 / Value: 206273
[System] [CHAT] Index: 13 / Value: 104003
[System] [CHAT] Index: 14 / Value: 59812
[System] [CHAT] Index: 15 / Value: 68538
[System] [CHAT] SORTED ARRAY (11 iterations):
[System] [CHAT] Index: 0 / Value: 1487
[System] [CHAT] Index: 1 / Value: 7529
[System] [CHAT] Index: 2 / Value: 16916
[System] [CHAT] Index: 3 / Value: 49401
[System] [CHAT] Index: 4 / Value: 59812
[System] [CHAT] Index: 5 / Value: 68538
[System] [CHAT] Index: 6 / Value: 93711
[System] [CHAT] Index: 7 / Value: 104003
[System] [CHAT] Index: 8 / Value: 112280
[System] [CHAT] Index: 9 / Value: 112477
[System] [CHAT] Index: 10 / Value: 122125
[System] [CHAT] Index: 11 / Value: 125808
[System] [CHAT] Index: 12 / Value: 168304
[System] [CHAT] Index: 13 / Value: 177818
[System] [CHAT] Index: 14 / Value: 195541
[System] [CHAT] Index: 15 / Value: 206273
```

## Datapack building
Building process of datapack:
1. Loading meta-data from `pack.mcmeta`
2. Loading std and scripts
3. Preprocessing (`main.c`)
4. Lexing/parsing C code and building AST by `pycparser` lib
5. Compilation/translation AST to MCF_ASM
6. Compilation MCF_ASM to MCFunction
7. Saving (linking) MCFunction code to datapack

## C
CMCFC uses the C89 (ANSI C) standard.

`void setup()` function calls on datapack loading, `void loop()` function calls every tick.

## MCF_ASM
The compilation uses intermediate MCF_ASM (minecraft pseudo-Assembler) code. There are no multi-level branches in this code, and only basic operations are used.
Commands:
1. `SET declcode value` - sets static value of variable (declcode)
2. `CAST to_declcode from_declcode` - casts varibale (from_declcode) type; new type variable is to_declcode
3. `EQUATING declcode value_declcode` - equates value of a variable (declcode) to another variable (value_declcode)
4. `OPERATION operand declcode value_declcode targetcode` - stores result of a math operation (operand) between two variables (declcode and value_declcode) to variable (targetcode)
5. `CONDITION condition_declcode true_function_declcode false_function_declcode` -  calls 'false' function (false_function_declcode) if condition_declcode is 0; otherwise calls 'true' function (true_function_declcode)
6. `CALL function_declcode` - calls function (function_declcode)
7. `RETURN declcode` - returns declcode
8. `ASM text asm_out` - MCFunction code (text) insertion; asm_out is function out declcode
9. `ADDRESS ptr_declcode declcode` - stores declcode address to ptr_declcode
10. `DEREFERENCE declcode ptr_declcode` - derederence ptr_declcode to declcode
11. `PTREQ ptr_declcode value_declcode` - reverse dereference (value of dynamic address) from declcode to ptr_declcode
