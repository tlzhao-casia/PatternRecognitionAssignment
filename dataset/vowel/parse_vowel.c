#include <stdio.h>

#include "./vowel.data"

int main(){
  FILE* f = fopen("vowel.data-txt", "wt");

  for (int s = 0; s < NO_SPEAKERS; ++s){
    for (int v = 0; v < NO_VOWELS; ++v){
      float* vowel = &voweldata[s][v][0];
      for (int i = 0; i < NO_INPUTS; ++i){
        fprintf(f, "%.3f,", vowel[i]);
      } 
      fprintf(f, "%d\n", v);
    }
  }

  fclose(f);

  return 0;
}
