CSV históricos são fornecidos pelo usuário, nunca baixados ou presumidos reais.
Use o esquema de data/demo/partials.csv. Inclua todos os municípios já no instante
inicial com contagens zero e seus atributos conhecidos naquele instante. Não use
participação final como expected_votes. timestamp deve conter fuso horário UTC.
Resultado final é um rótulo separado e só deve ser utilizado para avaliação.
Preços opcionais: timestamp,ask,depth,source; ask deve ser executável para quantity
inteira (preço constante até depth ou VWAP calculado externamente para o tamanho).
Preços indicativos/último negócio não servem para simular execução.
