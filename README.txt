1 - Descrição do Problema
    Classificação de imagens (o programa deve identificar se o animal na imagem é um cachorro ou gato)

2 - Justificativa das tecnicas utilizadas
    Utilizado o redimensionamento das imagens de treino em 64x64 para não impactar a performance do programa durante
    É utilizado o numpy para converter o dataset para arrays para facilitar a manipulação e usar o train_test_split
    As imagens são normalizadas em 0 e 1
    São utilizadas 7 camadas na rede neural e com a camada de saida com 2 neuronios (gato ou cachorro)
    Após o treino é exibido o gráfico para a e

3 - Etapas realizadas

    Acesso ao dataset, tratamento das imagens do dataset, conversão do dataset para arrays numpy, separação da porção
      de treino e teste, criação do modelo de rede neural, avaliaçao do conjunto do teste, exibiçao
        do gráfico do desempenho exibição das imagens com a previsão e tratamento das imagens.

4 - Tempo total gosto

    2h

5 - Dificuldades encontradas:

    Definir tamanho de imagem durante o treino para que não afete tanto a acuracia quanto o tempo de processamento
        do treinamento, aumento de acurácia sem afetar muito o tempo de processamento com os batches.