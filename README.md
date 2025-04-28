# DAA_Assignment_2
# Maximal Clique Algorithm - Instructions
## Website Link
https://sagarthomas24.github.io/Daa_assignmet_1/
## Prerequisites
Ensure you have the following installed:

- A C++ compiler (**g++** recommended)
- A terminal or command prompt

## Steps to Run the Program

### 1. Extract the Files
Unzip the provided zip file and open the extracted folder in **VS Code** or your preferred editor.

### 2. Compile the Program
Open a terminal in the extracted folder and run the following command to compile the C++ files with optimization:

For The Research Paper:  EXACT Method Algorithm.
```bash
 g++ -O3 -o algo1 algo1.cpp
```

For The Research Paper:  CoreEXACT Method Algorithm.
```bash
 g++ -O3 -o algo4 algo4.cpp
```



### 3. Run the Program
Once compiled, execute the program using:

```bash
./algo1
```

```bash
./algo4
```
on running the program the code will ask to specify which dataset 

For NetScience.txt
``` bash
NetScience.txt
```
For AS-733.txt
```bash
AS-733.txt
```
For CA-HepTh.txt
``` bash
CA-HepTh.txt
```

For Yeast.txt
``` bash
Yeast.txt
```


## Modifications on the Input File  

1. **Removed Comments**  
   - The dataset initially contained comments, which were removed to ensure clean parsing.  

2. **Standardized Format**  
   - The first line of the file represents the number of vertices (`n`), number of edges (`m`), size of clique (`h`).
   - Each subsequent line contains two integers representing an edge between two vertices.  

3. **Ensured Proper Parsing**  
   - The program reads `n` and `m` first and then correctly processes the `m` edges.  

4. **User-Defined Input File**  
   - The filename is provided by the user instead of being hardcoded.  

These modifications make the input processing more structured and flexible.

## Modifications on the Script
Have to mention the input file in the int main function.

## Contributions  

| Name                        | ID               | Contributions                                                                 |
|-----------------------------|-----------------|-------------------------------------------------------------------------------|
| Karingattil Sagar Thomas    | 2022A7PS0156H   | Helped in implementing algo 1 and algo 4                                                                                                       |
| Abhinav Chitturi            | 2022A7PS0064H   | Contributed to the algo1 code and made the website and helped in writing report                                                                |
| Dheeraj M P                 | 2022A7PS0006H   | Contributed to the algo4 code and made the website and helped in writing report                                                                |
| Pradyum Agarwal             | 2022A7PS0369H   | Helped in implementing algo 1 and algo 4                                                                                                       |

                                                                  



## Notes
- `-O3` enables compiler optimizations for better performance.
- Ensure your input files (if any) are in the correct format before running.

For any issues, check error messages and ensure dependencies are installed.
