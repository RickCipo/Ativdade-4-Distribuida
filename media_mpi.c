//Enrique Cipolla Martins - RA: 10427834
//Henrique Ferreira Marciano - RA: 10439797
//Pedro Gabriel Fernandes - RA: 10437465

//Compilar: mpicc -O2 -o media_mpi media_mpi.c
//Executar: mpirun -np 4 ./media_mpi 1000

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //validação do argumento N
    if (argc != 2 || atoi(argv[1]) <= 0) {
        if (rank == 0)
            fprintf(stderr, "Uso: mpirun -np <P> %s <N>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    int N = atoi(argv[1]);

    //setor local com N valores aleatórios em [0,1]
    srand((unsigned)time(NULL) + rank);
    float *vetor = malloc(N * sizeof(float));
    if (vetor == NULL) {
        fprintf(stderr, "[Processo %d] Erro de alocação\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    //soma local (double para reduzir erro de arredondamento)
    double soma_local = 0.0;
    for (int i = 0; i < N; i++) {
        vetor[i] = rand() / (float)RAND_MAX;
        soma_local += vetor[i];
    }
    double media_local = soma_local / N;

    //soma global no rank 0
    double soma_global = 0.0;
    MPI_Reduce(&soma_local, &soma_global, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);

    //coleta das somas e médias locais no rank 0 (impressão ordenada)
    double *somas = NULL, *medias = NULL;
    if (rank == 0) {
        somas  = malloc(size * sizeof(double));
        medias = malloc(size * sizeof(double));
    }
    MPI_Gather(&soma_local,  1, MPI_DOUBLE, somas,  1, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);
    MPI_Gather(&media_local, 1, MPI_DOUBLE, medias, 1, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        for (int p = 0; p < size; p++)
            printf("[Processo %d] Soma local: %.3f, Média local: %.4f\n",
                   p, somas[p], medias[p]);

        double media_global = soma_global / ((double)N * size);
        printf("\n[Soma global] %.3f\n", soma_global);
        printf("[Média global] %.4f\n", media_global);

        free(somas);
        free(medias);
    }

    free(vetor);
    MPI_Finalize();
    return 0;
}
