"""
Step 4: Aspect-Based Sentiment Analysis (ABSA) of Beer Characteristics (BC)

:author: Denis Eiras

Functions:
    - 
"""

import pandas as pd
import ast
from step import Step
from Prompt_AI import Prompt_AI
import re
import json
from datetime import datetime
import time as time_module
import os

class Step_4(Step):

    def __init__(self) -> None:
        super().__init__()

    def wait_for_interval(self, start: str, end: str):
        """
        This function waits for a specific interval of time.
        Used to run the prompts in a specific time window, cheaper hours
        
        Args:
            start (str): The start time in the format "HH:MM".
            end (str): The end time in the format "HH:MM".
        """
        start_time = datetime.strptime(start, "%H:%M").time()
        end_time = datetime.strptime(end, "%H:%M").time()

        while True:
            now = datetime.now().time()
            if start_time <= end_time:
                inside = start_time <= now <= end_time
            else:
                # interval crossing midnight (e.g. 22:00–06:00)
                inside = now >= start_time or now <= end_time
            if inside:
                print("waiting for running hours")
                time_module.sleep(30)
            else:
                return
        
    def step_4_1_get_prompt_zero_shot(self):
        """
        This function creates the Prompt ABSA zero-shot.
      
        Parameters:
            self (object): The object instance that contains the data.
        Returns:
            str: The prompt ABSA zero-shot.
        """
            
        print(f'Step 4.1 - "Prompt ABSA zero-shot" creation')
        prompt_sys = """ 
Você é um extrator de aspectos de cerveja. Do texto, extraia os ‘aspectos’ e a ‘categoria’ relacionados aos aspectos da cerveja. As categorias devem estar \
dentre os valores: ‘visual’, ‘aroma’, ‘sabor’, ‘amargor’, ‘álcool’ e ‘sensação na boca’. Extraia o ‘sentimento’ dentre os valores ‘muito negativo’, ‘negativo’, ‘neutro’, \
‘positivo’ ou ‘muito positivo’ para cada par aspecto/categoria. \
Regras:
- Dividir o aspecto na menor unidade possível. Por exemplo: "Espuma branca de média duração" deve gerar dois aspectos: "espuma branca" e "espuma de média duração", ambas categorias: "visual" \
- O entendimento sobre o sentimento deve considerer o sentimento relacionado para cada aspecto. Não considerar o entendimento do modelo pré-treinado ou o sentimento geral do texto. Usar "neutro" para aspectos que não possuem um sentimento relacionado. \
    - Exemplo: "Aroma cítrico" deve ser sentimento "neutro". "Bom aroma cítrico." indica um sentimento "positivo" para o aspecto. Outro exemplo: "muito Sabor de café": não indica um sentimento muito positivo, mas "sabor de café muito agradável" sim. \
    - Exceções: "espuma de boa retenção" e os adjetivos "refrescante", "cremosa", "balanceado", "equilibrado" sempre indicam um sentimento positivo. \
- Se houver expressãoes parecidas com "o sabor acompanha o aroma", copiar todos os aspectos da categoria "aroma" para a categoria "sabor", bem como o sentimento relacionado. \
Cada avaliação a ser avaliada está compreendida entre chaves. Cada item contém "index", que registra o índice da avaliação e "review_comment", que é o texto a ser avaliado. \
Não faça comentários, apenas gere a saída dos campos extraídos no formato a seguir: ['index','aspecto','categoria','sentimento'],\
"""
        return prompt_sys


    def step_4_1_get_prompt_few_shots(self, prompt_zero_shot: str, num_shots: int, use_all_BC: bool = True):
        """
        This function creates the Prompt ABSA few-shots based on the Prompt ABSA zero-shot.
        The reviews were selected manually from base step_4_1__base_for_prompts_selection.csv, considering good and bad reviews 
        for 4 main styles of beer, by experienced reviweres, and 2 reviews from newbies
        Parameters:
            self (object): The object instance that contains the data.
            prompt_zero_shot (str): The prompt ABSA zero-shot.
            num_shots (int): The number of shots to generate.
            use_all_BC (bool): Whether to use all beer characteristcs of each review.
        """
        
        print(f'Step 4.1 - "Prompt ABSA few-shots" creation')
                    
        # beer_style review_user review_num_reviews review_general_rate review_comment
        #
        # ***** Wibier
        #
        # - experienced - low rate
        # Bruno Sicchieri	531	1.1
        style1_exp_lowrate = """
"De coloração amarelada, turva. Espuma de difícil formação, altamente efervescente e sem duração. Bom aroma \
trazendo notas cítricas de laranja e semente de coentro. Na boca, início e final amargos e efervescentes, quanto ao sabor... horrível... \
agitei para capturar um pouco do fermento sedimentando no fundo e creio que foi meu erro... é difícil descrever, exceto a sensação de estar \
estragada... sabor de giz e terra. Carbonatação baixa. Corpo médio. Uma terrível [BJCP2015] 24A: Witbier. Poupe suas papilas gustativas... \
ou experimente por sua própria conta e risco. \
['0', 'cor do líquido amarelado', 'visual', 'neutro'], \
['0', 'cor do líquido turvo', 'visual', 'neutro'], \
['0', 'formação de espuma baixa', 'visual', 'neutro'], \
['0', 'espuma efervescente', 'visual', 'neutro'], \
['0', 'espuma pouco persistente', 'visual', 'negativo'], \
['0', 'notas cítricas de laranja', 'aroma', 'positivo'], \
['0', 'notas cítricas de semente de coentro', 'aroma', 'positivo'], \
['0', 'efervescente', 'sensação na boca', 'neutro'], \
['0', 'giz', 'sabor', 'muito negativo'], \
['0', 'terra', 'sabor', 'muito negativo'], \
['0', 'carbonatação baixa', 'sensação na boca', 'neutro'], \
['0', 'corpo médio', 'sensação na boca', 'neutro'] \
"
"""
        #
        # - experienced - high rate
        # Fabio Vieira	907	4.4	 
        style1_exp_highrate = """
"Temperatura de degustação: Cinco graus Celsius. Cor: Amarelo-palha medianamente turva. Creme: Média formação \
de creme branco que mantém uma fina camada persistente, deixando marcas no tumbler. Aroma: Cítrico com notas de limão, especiarias como coentro\
e pimenta, muito bom. Sabor: Maltado com cereais, frutado de limão e especiarias dominam os sentidos. O final do gole apresenta-se levemente \
amargo, levemente ácido e picante. O sabor cítrico do limão permanece por todo o gole, se prolongando no retrogosto, apresentando excelente \
drinkability e refrescância absurda! Excelente breja!! \
['0', 'cor do líquido amarelo-palha', 'visual', 'neutro'], \
['0', 'medianamente turva', 'visual', 'neutro'], \
['0', 'formação de espuma média', 'visual', 'neutro'], \
['0', 'cor da espuma branca', 'visual', 'neutro'], \
['0', 'espuma persistente', 'visual', 'positivo'], \
['0', 'cítrico de limão', 'aroma', 'muito positivo'], \
['0', 'coentro', 'aroma', 'muito positivo'], \
['0', 'pimenta', 'aroma', 'muito positivo'], \
['0', 'maltado com cereais', 'sabor', 'neutro'], \
['0', 'frutado de limão', 'sabor', 'neutro'], \
['0', 'especiarias', 'sabor', 'neutro'], \
['0', 'final levemente amargo', 'amargor', 'neutro'], \
['0', 'final ácido leve', 'sabor', 'neutro'], \
['0', 'final picante', 'sabor', 'neutro'], \
['0', 'drinkability alta', 'sensação na boca', 'muito positivo'], \
['0', 'refrescância alta', 'sensação na boca', 'muito positivo'] \
"
"""

        #
        #
        # ***** German Weizen
        # - experienced - low rate
        #  Jota Fanchin Queiroz	563	1.2	
        style2_exp_lowrate = """
"Uma weiss significativamente inferior ao padrão do estilo. E nem falo em comparação com as bávaras mas com a \
Eisenbahn por exemplo. Aparência: coloração dourada clara turva com creme de média formação e baixa persistência. Aroma: acanhado. Sabor: \
notas de banana e nada de cravo com um final doce demais. Estranho. Corpo: aguado até para pilsen que dirá weiss. Final: estranho, seco e \
curto. Conjunto: desequilibrado pelo excesso do doce e pelo descompassado do corpo e carbonatação. Drinkability baixa e refrescância \
comprometida. \
['0', 'cor do líquido dourado claro', 'visual', 'neutro'], \
['0', 'turva', 'visual', 'neutro'], \
['0', 'formação de espuma médio', 'visual', 'neutro'], \
['0', 'espuma de baixa persistência', 'visual', 'neutro'], \
['0', 'notas de banana', 'sabor', 'neutro'], \
['0', 'dulçor alto', 'sabor', 'negativo'], \
['0', 'corpo aguado', 'sensação na boca', 'negativo'], \
['0', 'final seco e curto', 'sensação na boca', 'negativo'], \
['0', 'drinkability baixa', 'sensação na boca', 'negativo'], \
['0', 'refrescância baixa', 'sensação na boca', 'negativo'] \
"
"""
        #
        # - experienced - high rate
        # Eduardo Guimarães Insta @cervascomedu	2380	4,4	
        style2_exp_highrate = """
"Apresentou coloração dourada com espuma branca de média formação e longa persistência. \
No aroma temos banana, cravo, mel, floral e pão doce. Na boca as notas permanecem, complementadas por cereais, herbal sutil e toques \
picantes. Tem corpo médio, carbonatação moderada e sensação refrescante. Excelente!
['0', 'cor do líquido dourado', 'visual', 'neutro'], \
['0', 'cor da espuma branca', 'visual', 'neutro'], \
['0', 'formação de espuma média', 'visual', 'neutro'], \
['0', 'espuma persistente', 'visual', 'positivo'], \
['0', 'banana', 'aroma', 'neutro'], \
['0', 'cravo', 'aroma', 'neutro'], \
['0', 'mel', 'aroma', 'neutro'], \
['0', 'floral', 'aroma', 'neutro'], \
['0', 'pão doce', 'aroma', 'neutro'], \
['0', 'banana', 'sabor', 'neutro'], \
['0', 'cravo', 'sabor', 'neutro'], \
['0', 'mel', 'sabor', 'neutro'], \
['0', 'floral', 'sabor', 'neutro'], \
['0', 'pão doce', 'sabor', 'neutro'], \
['0', 'cereais', 'sabor', 'neutro'], \
['0', 'herbal sutil', 'sabor', 'neutro'], \
['0', 'notas picantes', 'sabor', 'neutro'], \
['0', 'corpo médio', 'sensação na boca', 'neutro'], \
['0', 'carbonatação moderada', 'sensação na boca', 'neutro'], \
['0', 'refrescância alta', 'sensação na boca', 'positivo'] \
"
"""
        #
        #
        # ***** India Pale Ale (IPA)
        # - experienced - low rate
        # Wagner Gasparetto	700	1,5	
        style3_exp_lowrate = """
"Cor amarela clara, com certa turbidez, de cara fugindo um pouco da expectativa do estilo. Aroma maltado com \
cítrico muito suave e paladar maltado, pouco lupulado e quase sem presença cítrica. Longe de uma IPA. Média carbonatação e boa drinkability,\
corpo leve. Desagradou.... \
['0', 'cor do líquido amarelo claro', 'visual', 'negativo'], \
['0', 'líquido turvo', 'visual', 'negativo'], \
['0', 'maltado', 'aroma', 'neutro'], \
['0', 'cítrico muito suave', 'aroma', 'neutro'], \
['0', 'maltado', 'sabor', 'neutro'], \
['0', 'pouco lupulado', 'sabor', 'neutro'], \
['0', 'cítrico muito suave, 'sabor', 'neutro'], \
['0', 'média carbonatação', 'sensação na boca', 'neutro'], \
['0', 'drinkability boa', 'sensação na boca', 'positivo'], \
['0', 'corpo baixo', 'sensação na boca', 'neutro'] \
"
"""
        #
        # - experienced - high rate
        # Alexandre LC	571	4,7	
        style3_exp_highrate = """
"Pataqueparéu, não sei o que dizer sobre esta cerveja! Sorvida e provada logo em seguida a perigosa. Coloração âmbar\
alaranjada. Espuma levemente bege, com alta formação e boa duração. Apesar da tampinha ser o mesmo problema que a Perigosa, como foi bem \
linda no copo leva 5/5 em aparência. Aroma é fodástico, aparecendo com um buquê fenomenal. Percepção floral, cítrica, caramelada, de melaço\
e de chocolate cremoso (lembra muito o GALAK®). Com notas herbais e de laranja ao fundo. Um conjunto bem equilibrado e perfeito. \
Perfumadíssima. Aroma pra mim é 6/5! kkkk Sabor é inicialmente doce, doce de chocolate cremoso/branco, cacau, caramelo/toffe, logo mesclado\
com um amargor leve e um malte torrado bem sutil. Corpo denso e licoroso. Conjunto equilibrado e primoroso, no qual o doce inicial se acerta\
e abraça bem o amargor floral final. Final seco e levemente amargo. Retrogosto amargo e denso. SENSACIONAL. É uma IPA diferente, devido ao \
fato de o seu padrão puxar muito mais pro doce do que pro amargor lupulento, não compararei com as demais IPAs, pra mim entraria como uma \
Specialty Beer. Já está entre as minhas favoritas. Mais um preço abusivo da Bodebrown... quase R$7 por 100mL. Vacilo. \
['0', 'cor do líquido âmbar alaranjado', 'visual', 'muito positivo'], \
['0', 'cor da espuma bege leve', 'visual', 'muito positivo'], \
['0', 'formação de espuma alta', 'visual', 'muito positivo'], \
['0', 'espuma persistente', 'visual', 'muito positivo'], \
['0', 'floral', 'aroma', 'muito positivo'], \
['0', 'cítrico', 'aroma', 'muito positivo'], \
['0', 'caramelado', 'aroma', 'muito positivo'], \
['0', 'melaço', 'aroma', 'muito positivo'], \
['0', 'chocolate cremoso', 'aroma', 'muito positivo'], \
['0', 'herbal', 'aroma', 'muito positivo'], \
['0', 'laranja', 'aroma', 'muito positivo'], \
['0', 'dulçor', 'sabor', 'neutro'], \
['0', 'doce de chocolate branco/cremoso', 'sabor', 'neutro'], \
['0', 'cacau', 'sabor', 'positivo'], \
['0', 'caramelo/toffe', 'sabor', 'neutro'], \
['0', 'amargor floral', 'amargor', 'neutro'], \
['0', 'malte torrado leve', 'sabor', 'neutro'], \
['0', 'corpo denso', 'sensação na boca', 'neutro'], \
['0', 'corpo licoroso', 'sensação na boca', 'positivo'], \
['0', 'final seco', 'sensação na boca', 'neutro'] \
['0', 'amargor leve', 'amargor', 'neutro'], \
['0', 'retrogosto amargo', 'amargor', 'neutro'], \
"
"""
        #  ONE SHOT EXAMPLE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        # - inexperienced - low rate
        # Thiago Coelho	1	1,5	
        style3_inexp_lowrate = """
"Rótulo agradável, em garrafa âmbar bojuda. Tampa sem rótulo, dando um aspecto desleixado à cerveja. As cervejas bastante lupuladas sempre têm uma agradável \
antecipação do aroma logo quando se abre a garrafa. Essa não tinha: mau presságio... Cor âmbar, translúcida,excelente sensação visual ao ser servida, particularmente \
pela intensa formação de espuma, que é persistente. Aroma herbáceo, suave demais,muito aquém para uma cerveja que carrega no lúpulo aromático, inclusive tendo sido  \
feito dry hopping com Cascade, um lúpulo essencialmente aromático . Se a intenção era fazer uma autêntica American IPA, vê-se aqui mais pra uma do velho continente,  \
inglesa, precisamente. No sabor,perde-se completamente: tem um amargor intenso mas adstringente, incomodativo, que parece mesmo arranhar a língua e que perdura no  \
aftertaste.Baixíssima drinkability. Vê-se muito malte, particularmente no aftertaste, quando se mantém um retrogosto de mel e pão. Não vi qualquer off-flavor no exemplar \
que degustei. Pergunto: é uma IPA ou é uma American Pale Ale lupulada em excesso (amargor excessivo, por ser incomodativo...)? \
['0', 'cor do líquido âmbar', 'visual', 'neutro'], \
['0', 'líquido translúcido', 'visual', 'neutro'], \
['0', 'formação de espuma ótima', 'visual', 'muito positivo'], \
['0', 'espuma persistente', 'visual', 'muito positivo'], \
['0', 'herbáceo suave demais', 'aroma', 'negativo'], \
['0', 'amargor intenso', 'amargor', 'negativo'], \
['0', 'adstringente', 'amargor', 'negativo'], \
['0', 'baixíssima drinkability', 'sensação na boca', 'muito negativo'], \
['0', 'maltado alto', 'sabor', 'neutro'], \
['0', 'retrogosto de mel', 'sabor', 'neutro'], \
['0', 'retrogosto de pão', 'sabor', 'neutro'], \
['0', 'sem off-flavor', 'aroma', 'positivo'], \
['0', 'lupulada em excesso', 'sabor', 'negativo'] \
"
"""
        #
        # - inexperienced - high rate
        # Odonio dos Anjos Filho	1	4,7	
        style3_inexp_highrate = """
"Cerveja com sabor de cerveja forte. Lúpulo e álcool presentes que dão o perfeito sabor de cerveja India \
Palle Ale. Mais fantástico ainda reconhecer uma cerveja dessa no Brasil, respeitando os processos de pureza necessários para fabricação de \
grandes cervejas. Vale tomar com comidas mais fortes e apreciar durante todo o ano. Espuma maravilhos que matém o aroma da cerveja de forma \
prolongada. Uma perfeição em termos de equilíbrio. Sensacional! \
['0', 'formação de espuma boa', 'visual', 'muito positivo'] \
"
"""
        #
        # ***** Porter
        # - experienced - low rate
        # Alexandre LC	571	1,7	
        style4_exp_lowrate = """
"Coloração negra opaca. Espuma bege de alta formação e pouca duração. Aroma de caramelo e açúcar mascavo. Sabor quase \
exclusivo de caramelo, com leve torrado e um dulçor muito acima da média, enjoativa demais. Praticamente uma malzbier menos doce. Totalmente \
fora do estilo. Bebi apenas um copo e deixei o resto para mulherada. \
['0', 'cor do líquido negro opaca', 'visual', 'neutro'], \
['0', 'cor da espuma bege', 'visual', 'neutro'], \
['0', 'formação de espuma alta', 'visual', 'neutro'], \
['0', 'espuma pouco persistente', 'visual', 'negativo'], \
['0', 'caramelo', 'aroma', 'neutro'], \
['0', 'ácúcar mascavo', 'aroma', 'neutro'], \
['0', 'caramelo', 'sabor', 'neutro'], \
['0', 'torrado leve', 'sabor', 'neutro'], \
['0', 'dulçor alto', 'sabor', 'negativo'] \
"
"""
        #
        # - experienced - high rate
        # Odimi Toge	1031	4,6	
        style4_exp_highrate = """
"Bebida desenvolvida em parceria com a Cachaçaria Nacional - maior varejista de cachaças do mundo, sediada em Belo \
Horizonte (MG).  Trata-se de um blend de Baltic Porter com a cachaça Legítima de Minas, na proporção de 10%.  Envelhecida por dois anos em \
barris de amburana, esta cachaça é produzida em Itaverava (MG) no Alambique Taverna de Minas.  A receita toda, criada pelo cervejeiro caseiro \
Fábio Ferreira, foi medalha de Ouro do XII Concurso da Acerva Mineira.  Aroma intenso de cachaça, passando por coco, canela, baunilha e mel. \
Toffee, melaço e ameixa seca surgem sinérgicos. Espetáculo! Líquido castanho avermelhado, permitindo certa passagem de luz. Servido, forma uma \
camada fina e efêmera de espuma bege clara. Na boca mostra corpo médio e reduzida carbonatação. A junção de cachaça e cerveja conversa bem, \
resultando em notas de coco queimado, canela, baunilha, ameixa seca e café - riscadas por leve dulçor maltado. Álcool inacreditavelmente bem \
inserido (sério, cadê esse álcool todo anunciado?) O final segue ligeiramente adocicado, com bastante cachaça e breve torrado.  "Drinkability" \
relativamente alta em vista de toda sua "periculosidade", por assim dizer.  Blend muito bem construído, com cerveja e cachaça na mais perfeita \
harmonia. Parabéns aos envolvidos! ???? \
['0', 'intenso de cachaça', 'aroma', 'muito positivo'], \
['0', 'coco', 'aroma', 'muito positivo'], \
['0', 'canela', 'aroma', 'muito positivo'], \
['0', 'baunilha', 'aroma', 'muito positivo'], \
['0', 'mel', 'aroma', 'muito positivo'], \
['0', 'toffee', 'aroma', 'muito positivo'], \
['0', 'melaço', 'aroma', 'muito positivo'], \
['0', 'ameixa seca', 'aroma', 'muito positivo'], \
['0', 'cor do líquido castanho avermelhado', 'visual', 'neutro'], \
['0', 'cor do líquido semi translúcido', 'visual', 'neutro'], \
['0', 'cor da espuma bege clara ', 'visual', 'neutro'], \
['0', 'corpo médio', 'sensação na boca', 'neutro'], \
['0', 'carbonatação baixa', 'sensação na boca', 'neutro'], \
['0', 'coco queimado', 'sabor', 'positivo'], \
['0', 'canela', 'sabor', 'positivo'], \
['0', 'baunilha', 'sabor', 'positivo'], \
['0', 'ameixa seca', 'sabor', 'positivo'], \
['0', 'café', 'sabor', 'positivo'], \
['0', 'dulçor maltado leve', 'sabor', 'positivo'], \
['0', 'alcool muito bem inserido', 'alcool', 'muito positivo'], \
['0', 'final com dulçor leve', 'sabor', 'neutro'], \
['0', 'final com bastante cachaça', 'sabor', 'neutro'], \
['0', 'final leve torrado', 'sabor', 'neutro'], \
['0', 'drinkability relativamente alta', 'sensação na boca', 'positivo'] \
"
"""

        prompt_few_shots = prompt_zero_shot + """ \
Abaixo, entre aspas, exemplos de textos de avaliações e o resultado esperado. \
Ignore o valor do campo index dos exemplos, pois são apenas para mostrar o formato de saída.
"""
   
        if use_all_BC:
            if num_shots == 1:
                prompt_few_shots += style3_inexp_lowrate
            
            elif num_shots == 3:
                prompt_few_shots += style3_inexp_lowrate
                
                prompt_few_shots += style1_exp_lowrate
                prompt_few_shots += style2_exp_highrate
            elif num_shots == 10:
                prompt_few_shots += style3_inexp_lowrate
                
                prompt_few_shots += style1_exp_lowrate
                prompt_few_shots += style1_exp_highrate
                prompt_few_shots += style2_exp_lowrate
                prompt_few_shots += style2_exp_highrate
                prompt_few_shots += style3_exp_lowrate
                prompt_few_shots += style3_exp_highrate
                prompt_few_shots += style3_inexp_highrate
                prompt_few_shots += style4_exp_lowrate
                prompt_few_shots += style4_exp_highrate

        else:
        
            # - inexperienced - low rate
            # Thiago Coelho	1	1,5	
            style3_inexp_lowrate_1_CC = """
"Rótulo agradável, em garrafa âmbar bojuda. Tampa sem rótulo, dando um aspecto desleixado à cerveja. As cervejas bastante lupuladas sempre têm uma agradável \
antecipação do aroma logo quando se abre a garrafa. Essa não tinha: mau presságio... Cor âmbar, translúcida,excelente sensação visual ao ser servida, particularmente \
pela intensa formação de espuma, que é persistente. Aroma herbáceo, suave demais,muito aquém para uma cerveja que carrega no lúpulo aromático, inclusive tendo sido  \
feito dry hopping com Cascade, um lúpulo essencialmente aromático . Se a intenção era fazer uma autêntica American IPA, vê-se aqui mais pra uma do velho continente,  \
inglesa, precisamente. No sabor,perde-se completamente: tem um amargor intenso mas adstringente, incomodativo, que parece mesmo arranhar a língua e que perdura no  \
aftertaste.Baixíssima drinkability. Vê-se muito malte, particularmente no aftertaste, quando se mantém um retrogosto de mel e pão. Não vi qualquer off-flavor no exemplar \
que degustei. Pergunto: é uma IPA ou é uma American Pale Ale lupulada em excesso (amargor excessivo, por ser incomodativo...)? 
"""

            if num_shots == 1:
                prompt_few_shots += style3_inexp_lowrate_1_CC
                prompt_few_shots += """\
['0', 'cor do líquido âmbar', 'visual', 'neutro'], \
"
"""
            elif num_shots == 3:
                prompt_few_shots += style3_inexp_lowrate_1_CC
                prompt_few_shots += """\
['0', 'cor do líquido âmbar', 'visual', 'neutro'], \
['0', 'líquido translúcido', 'visual', 'neutro'], \
['0', 'formação de espuma ótima', 'visual', 'muito positivo'] \
"
"""

        return prompt_few_shots

    def run_step_4_1_create_base_prompts(self):
        """
        This function selects reviews for creating the "Prompts Shots".
        To construct the one-shot and few-shot prompts, 10 reviews were pre-selected from the "Reviews Main" base.
        The selection ensured diversity across four beer styles from different BJCP style categories, balanced 
        between positive and negative evaluations and drawn from both experienced and novice reviewers
        
        Parameters:
            self (object): The object instance that contains the data.
        Returns:
            pandas.DataFrame: A DataFrame containing the selected reviews.
        """

        print(f'Step 4.1 - Selections of reviews for Prompts ABSA')
        styles_for_prompt = ['India Pale Ale (IPA)', 'German Weizen', 'Porter', 'Witbier']
        pre_selected_reviews = self.inp_out_df[ self.inp_out_df['beer_style'].isin(styles_for_prompt) & 
                                     ((self.inp_out_df['review_general_rate'] >= 4) | (self.inp_out_df['review_general_rate'] <= 2)) & 
                                     ((self.inp_out_df['review_num_reviews'] >= 368) | (self.inp_out_df['review_num_reviews'] == 1))]
        pre_selected_reviews = pre_selected_reviews.sort_values(by=['beer_style', 'review_general_rate', 'review_num_reviews'])
        
        # df is the initial base for selection of reviews for prompts
        
        # Then do the manual selection of reviews from the previous pre_selected_reviews DataFrame
        selected_review_comments_starting_with_the_strings = [
            "De coloração amarelada, turva. Espuma de difícil formação, altamente efervescente e sem duração. Bom aroma",
            "Temperatura de degustação: Cinco graus Celsius. Cor: Amarelo-palha medianamente turva. Creme: Média formação",
            "Uma weiss significativamente inferior ao padrão do estilo. E nem falo em comparação com as bávaras mas com a",
            "Apresentou coloração dourada com espuma branca de média formação e longa persistência.",
            "Cor amarela clara, com certa turbidez, de cara fugindo um pouco da expectativa do estilo. Aroma maltado com",
            "Pataqueparéu, não sei o que dizer sobre esta cerveja! Sorvida e provada logo em seguida a perigosa. Coloração âmbar",
            "Rótulo agradável, em garrafa âmbar bojuda. Tampa sem rótulo, dando um aspecto desleixado à cerveja. As cervejas bastante lupuladas sempre têm uma agradável",
            "Cerveja com sabor de cerveja forte. Lúpulo e álcool presentes que dão o perfeito sabor de cerveja India",
            "Coloração negra opaca. Espuma bege de alta formação e pouca duração. Aroma de caramelo e açúcar mascavo. Sabor quase",
            "Bebida desenvolvida em parceria com a Cachaçaria Nacional - maior varejista de cachaças do mundo, sediada em Belo"
        ]
        
        manual_selected_df = pd.DataFrame()
        for starting_string in selected_review_comments_starting_with_the_strings:
            selected_row = pre_selected_reviews[pre_selected_reviews['review_comment'].str.startswith(starting_string)].head(1)
            manual_selected_df = pd.concat([manual_selected_df, selected_row], ignore_index=True)
        
        manual_selected_df.to_csv(f'{self.work_dir}/step_4_1__base_for_prompts_selection.csv')
        
        return manual_selected_df
    
    def run_step_4_1_create_base_reviews_sample(self, reviews_for_prompts_df: pd.DataFrame):
        """
        This function selects reviews for Prompt ABSA based on certain criteria.
        Creates the "Base Reviews Sample", for testing prompts zero, one and few shots.
        
        Parameters:
            self (object): The object instance that contains the data.
        Returns:
            pandas.DataFrame: A DataFrame containing the selected reviews. The DataFrame is sorted by beer style, 
            review general rate, and review number of reviews.
        """            
       
        # Base Prompts creation
        print(f'Step 4.1 - Base Prompts Validation: creating')
        print(f'- Initial line count: {len(self.inp_out_df)}')
        df = self.inp_out_df

        # remove registers of df containing reviews_for_prompts_df registers, to not influence on validation
        df = df[~df.isin(reviews_for_prompts_df)]
        print(f'- Removing registers from reviews_for_prompts_df - Parcial line count (Base Prompts): {len(df)}')
        
        # review_comment size >= 75% of the greatest sizes 
        greatest_review_comment_size_threshold = df['review_comment_size'].quantile(0.75) 
        df = df[df['review_comment_size'] >= greatest_review_comment_size_threshold]
        print(f'- Select greatest review comment size (quantile 75%) - Parcial line count (Base Prompts): {len(df)}')
        
        
        # group the reviews by year
        df['review_year'] = pd.to_datetime(df['review_datetime']).dt.year
        df = df.groupby('review_year')
        
        # for each group if year, filter: 
        for year in df.groups.keys():
            df_year = df.get_group(year)
            print(f'\nProcessing year: {year} - Initial line count: {len(df_year)}')
            
            # select only one register per review_user column on df;
            df_year = df_year.groupby('review_user').head(1)
            print(f'- Select maximum of one reviews per user - Parcial line count (Base Prompts): {len(df_year)}')
            
            # Best and worst reviews
            df_year = df_year[(df_year['review_general_rate'] >= 3) | (df_year['review_general_rate'] <= 2.0)]
            print(f'- Select best and worst reviews review_general_rate<=2 or >=3 - Parcial line count (Base Prompts): {len(df_year)}')
            
            # select max 5 registers per beer_style column on df
            df_year = df_year.groupby('beer_style').head(1)
            print(f'- Select maxiumum of reviews per beer_style - Parcial line count (Base Prompts): {len(df_year)}')
        
            # # percentil 2% reviwers with most reviews “review_num_reviews” and inexperient (<=2 reviews)
            # greatest_reviewers_threshold = df_year['review_num_reviews'].quantile(0.90)
            # df_year = df_year[ (df_year['review_num_reviews'] >= greatest_reviewers_threshold) | (df_year['review_num_reviews'] <= 10)]
            # print(f'- Select experients (quantile 2%) and inexperients (<=2 reviews) - Parcial line count (Base Prompts): {len(df_year)}')  
            
            if 'df_final' in locals():
                df_final = pd.concat([df_final, df_year], ignore_index=True)
            else:
                df_final = df_year.copy()
                
        df = df_final.copy()
            
        # select maximum of 6 reviews by year
        df = df.groupby('review_year').head(6)
        print(f'\nSelect maximum of 6 reviews by year - Parcial line count (Base Prompts): {len(df)}')
        
        df = df.sort_values(by=['review_datetime', 'review_general_rate', 'review_num_reviews'])
        print(f'\nFinal line count (Base Prompts Evaluation): {len(df)}')
        
        df.to_csv(f'{self.work_dir}/step_4_1__reviews_sample.csv', index=False)
        return df

    def llm_batch_equivalence_judge(self, pred_df, gold_df, error_count, prompt_ai):
        """
        Iterates over aspects annotated (gold), calling the LLM once per gold.
        Each call receives 1 gold and all preds not yet used.
        Produces pairs one-to-one and removes pairs already matched.
        
        Parameters:
            pred_df (pandas.DataFrame): The DataFrame containing the predictions.
            gold_df (pandas.DataFrame): The DataFrame containing the gold annotations.
            error_count (int): The current error count.
            prompt_ai (Prompt_AI): The Prompt_AI object.
        """

        pred_list = [
            {
                "id": i,
                "aspect": r["aspect"],
                "category": r["category"],
                "sentiment": r["sentiment"],
            }
            for i, r in pred_df.reset_index(drop=True).iterrows()
        ]

        gold_list = [
            {
                "id": i,
                "aspect": r["aspect"],
                "category": r["category"],
                "sentiment": r["sentiment"],
            }
            for i, r in gold_df.reset_index(drop=True).iterrows()
        ]

        remaining_preds = pred_list.copy()
        matches = []

        for gold in gold_list:

            if not remaining_preds:
                matches.append(
                    {
                        "gold_id": gold["id"],
                        "pred_id": -1,
                        "aspect_ok": False,
                        "category_ok": False,
                        "sentiment_ok": False,
                    }
                )
                continue

            gold_item = {
                "id": gold["id"],
                "aspect": gold["aspect"],
                "category": gold["category"],
            }

            pred_items = [
                {
                    "id": p["id"],
                    "aspect": p["aspect"],
                    "category": p["category"],
                }
                for p in remaining_preds
            ]

            prompt = f"""
    Você é um avaliador de equivalência semântica para ABSA.

    TAREFA:
    Compare o ASPECTO ANOTADO com os ASPECTOS PREVISTOS e identifique
    no máximo UM pareamento semântico.

    RETORNE APENAS um JSON válido no formato:

    {{
        "gold_id": <int>,
        "pred_id": <int | -1>,
        "aspect_ok": <true|false>,
        "category_ok": <true|false>
    }}

    REGRAS:
    - Use somente os ids fornecidos.
    - Se NÃO houver aspecto previsto semanticamente equivalente:
        - aspect_ok = false
        - pred_id = -1
        - category_ok = false
    - Se houver aspecto semanticamente equivalente:
        - aspect_ok = true
        - pred_id = id correspondente
        - category_ok = true SOMENTE se as categorias forem semanticamente equivalentes.

    ASPECTO ANOTADO:
    {gold_item}

    ASPECTOS PREVISTOS:
    {pred_items}
    """
            prompt_ai.prompt = prompt
            response, finish_reason = prompt_ai.get_completion()

            if finish_reason != "stop":
                error_count += 1
                print(f'Error count: {error_count}')
                continue
            
            try:
                response = response.replace("```json", "").replace("```", "").strip()
                match = json.loads(response)
            except Exception:
                print(f'Error parsing response: {response}')
                error_count += 1
                print(f'Error count: {error_count}')
                continue

            # Pós-processamento mínimo de segurança
            match["sentiment_ok"] = False

            if match["aspect_ok"] is True and match["pred_id"] != -1:
                pred_id = match["pred_id"]

                pred_item = next(
                    (p for p in remaining_preds if p["id"] == pred_id), None
                )

                if pred_item is not None:
                    # Remove pred utilizado
                    remaining_preds = [
                        p for p in remaining_preds if p["id"] != pred_id
                    ]

                    # Avaliação determinística de sentimento
                    if pred_item["sentiment"] == gold["sentiment"]:
                        match["sentiment_ok"] = True

            matches.append(match)

        return matches, error_count

    def run_step_4_2_ABSA_model_shots_evaluation(self, df_reviews_sample):
        """
        Runs ABSA of vary models and shots on Reviews Sample.
        Compares against step_4_ABSA_Gold.csv and computes macro Precision / Recall / F1.
        
        Parameters:
            df_reviews_sample (pandas.DataFrame): The DataFrame containing the base prompts.
        """

        annotated_file = f"{self.work_dir}/step_4_ABSA_Gold.csv"
        df_gold = pd.read_csv(annotated_file, sep=",", encoding="utf-8")
        
        # GPT was selected for this simpler task due to best results on the test set
        prompt_ai = Prompt_AI("gpt-4o-mini", None)

        num_reviews_to_process = len(df_gold) + 1

        # for reviews_per_request in [1, 3, 6, ,9 ,12]:
        from itertools import product

        models = ['sabia-3', 'gpt-4o-mini']
        use_all_BC_opts = [True, False]
        nshots_opts = [0, 1, 3]
        reviews_per_request_opts = [1]
        # reviews_per_request_opts = [1, 3, 6, 9, 12]
        wait_for_interval = False
        
 
        for model, use_all_BC, nshots, reviews_per_request in product(models, use_all_BC_opts, nshots_opts, reviews_per_request_opts):
            print(f"Using {model} with {nshots} shots and {use_all_BC} BC, for {reviews_per_request} reviews per request")
    
            if nshots == 0 and use_all_BC == True:  
                continue
            
            file_basename=f'{self.work_dir}/step_4_2__{nshots}shots_{model}_{"all_BC" if use_all_BC else f"{nshots}_BC"}_{reviews_per_request}rev_per_req'
            error_count = 0
            
            n_shot_file_name = f'{file_basename}_from_0.csv'

            # uncomment to not process again - run_ABSA ignores processed items
            # if os.path.exists(n_shot_file_name):
            #     df_pred = pd.read_csv(n_shot_file_name, sep=",", encoding="utf-8")
            #     df_pred['index'] = pd.to_numeric(df_pred['index'], errors='coerce')
            #     print(f'\n\n****************************\ndf_pred - line count: {len(df_pred)} \n\n')
            # else:
            
            df_pred, n_shot_file_name = self.run_ABSA_parallel(
                'step_4_2',
                df_reviews_sample,
                model,
                nshots,
                reviews_per_request,
                num_reviews_to_process=num_reviews_to_process,
                use_all_BC=use_all_BC,
                wait_for_interval = wait_for_interval
            )
            
            
            df_scores_filename = f'{file_basename}_scores.csv'
            try:
                print(f'Reading {df_scores_filename}')
                df_scores = pd.read_csv(df_scores_filename, sep=",", encoding="utf-8")
                # per_review_scores is a list of dicts
                per_review_scores = df_scores.to_dict('records')
            except:
                print(f'{df_scores_filename} not exists')
                df_scores = None
                per_review_scores = []

            for idx in df_gold['index'].unique():
                
                gold_i = df_gold[df_gold['index'] == idx]
                pred_i = df_pred[df_pred['index'] == idx]

                # ignore non existing reviews in the validation set or in the predicted set
                if len(gold_i) == 0:
                    print(f'No gold reviews found for review {idx} !!! Skipping')
                    continue
                
                if len(pred_i) == 0:
                    print(f'No reviews found for review {idx} !!!')
                    a_correct = 0
                    b_correct = 0
                    c_correct = 0
                    a_total_pred = 0
                else:
                    if df_scores is not None and idx in df_scores['index'].unique():
                        print(f'Found review {idx} in df_scores, skipping')
                        continue

                    matches, error_count = self.llm_batch_equivalence_judge(pred_i, gold_i, error_count, prompt_ai)
                    if len(matches) == 0:
                        print(f'No matches found for review {idx}')
                        continue
                        
                    print("matches", matches)

                    a_correct = sum(1 for m in matches if m["aspect_ok"])
                    b_correct = sum(1 for m in matches if m["aspect_ok"] and m["category_ok"])
                    c_correct = sum(
                        1 for m in matches
                        if m["aspect_ok"] and m["category_ok"] and m["sentiment_ok"]
                    )

                    a_total_pred = len(pred_i)
                    a_correct = min(a_correct, a_total_pred)
                    b_correct = min(b_correct, a_total_pred)
                    c_correct = min(c_correct, a_total_pred)
                
                a_total_gold = len(gold_i)    
                print("a_total_pred", a_total_pred)
                print("a_total_gold", a_total_gold)
                print("a_correct", a_correct)
                print("b_correct", b_correct)
                print("c_correct", c_correct)

                per_review_scores.append({
                    "index": idx,
                    "a_correct": a_correct,
                    "b_correct": b_correct,
                    "c_correct": c_correct,
                    "a_total_pred": a_total_pred,
                    "a_total_gold": a_total_gold,
                })
                
                df_scores = pd.DataFrame(per_review_scores)
                df_scores.to_csv(df_scores_filename, index=False)

            # write final results to csv
            results = []
            
            def preci_recall(pred_correct, total_pred_or_gold):
                return pred_correct / total_pred_or_gold if total_pred_or_gold > 0 else 0
                    
            def f1(prec, recall):
                return 2*prec*recall/(prec+recall) if (prec+recall) > 0 else 0
                    
            for model, use_all_BC, nshots, reviews_per_request in product(models, use_all_BC_opts, nshots_opts, reviews_per_request_opts):
            
                # skips BC if nshots == 0
                if nshots == 0 and use_all_BC == True:  
                    continue
                
                file_basename=f'{self.work_dir}/step_4_2__{nshots}shots_{model}_{"all_BC" if use_all_BC else f"{nshots}_BC"}_{reviews_per_request}rev_per_req'
                df_scores_filename = f'{file_basename}_scores.csv'
                
                try:
                    df_scores = pd.read_csv(df_scores_filename, sep=",", encoding="utf-8")
                except:
                    print(f'File not found: {df_scores_filename} , skipping evaluation metrics')
                    continue
                
                # print("df_scores", df_scores)
                a_correct_total = df_scores["a_correct"].sum()
                b_correct_total = df_scores["b_correct"].sum()
                c_correct_total = df_scores["c_correct"].sum()
                a_total_gold_total = df_scores["a_total_gold"].sum()
                a_total_pred_total = df_scores["a_total_pred"].sum()
                
                a_prec = preci_recall(a_correct_total, a_total_pred_total)
                b_prec = preci_recall(b_correct_total, a_total_pred_total)
                c_prec = preci_recall(c_correct_total, a_total_pred_total)
                
                a_rec = preci_recall(a_correct_total, a_total_gold_total)
                b_rec = preci_recall(b_correct_total, a_total_gold_total)
                c_rec = preci_recall(c_correct_total, a_total_gold_total)
                
                a_f1 = f1(a_prec, a_rec)
                b_f1 = f1(b_prec, b_rec)
                c_f1 = f1(c_prec, c_rec)
            
                metrics = {
                    "model": model,
                    "nshots": nshots,
                    "use_all_BC": use_all_BC,
                    "a_prec": a_prec,
                    "b_prec": b_prec,
                    "c_prec": c_prec,
                    "a_rec": a_rec,
                    "b_rec": b_rec,
                    "c_rec": c_rec,
                    "a_f1": a_f1,
                    "b_f1": b_f1,
                    "c_f1": c_f1,
                }
                results.append(metrics)

            df_results = pd.DataFrame(results)
            df_results_filename = f'{self.work_dir}/step_4_2____evaluation_metrics_{reviews_per_request}rev_per_req.csv'
            df_results.to_csv(df_results_filename, index=False)

            print("\nStep 4.2 evaluation completed")
            print(df_results)          
                
    def run_step_4_3_evaluate_main_base(self):
        """
        The best model and number of shots runs on the Reviews Main base.
        
        Parameters:
            self (object): The object instance that contains the data.
        Returns:
            absa_main_df (pandas.DataFrame): The DataFrame containing the ABSA Main base.
        
        """
            
        best_model = 'sabia-3'
        best_nshots = 3
        reviews_per_request = 9
        num_reviews_to_process = 10e6
        use_all_BC = True
        wait_for_interval = True
        max_concurrent_batches = 500
        submit_delay_sec = 0.1  # PromptAI with 0.1 delay
        
        # print date and time
        init_time = datetime.now()
        print(f'Running Step 4.3 ABSA on Main Base at {init_time.strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'- df_main_base - line count: {len(self.inp_out_df)}')
        absa_main_df, _ = self.run_ABSA_parallel('step_4_3', self.inp_out_df, best_model, best_nshots, 
                      reviews_per_request=reviews_per_request, num_reviews_to_process=num_reviews_to_process, use_all_BC = use_all_BC, wait_for_interval = wait_for_interval, max_concurrent_batches = max_concurrent_batches, submit_delay_sec = submit_delay_sec)
        
        end_time = datetime.now()
        print(f"\nStep 4.3 ABSA on Main Base completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total running time: {end_time - init_time}")
        print(f'- absa_main_df - line count: {len(absa_main_df)}')
        return absa_main_df

    def run_ABSA_parallel(self, step_name, df_base, model, nshots, reviews_per_request = 10, num_reviews_to_process = None, use_all_BC = True, wait_for_interval = True, max_concurrent_batches = 3, submit_delay_sec = 0.5):
        """
        Parallel version of run_ABSA that processes multiple batches concurrently.

        Parameters:
            step_name (str): The name of the step.
            df_base (pandas.DataFrame): The DataFrame containing the base.
            model (str): The name of the model.
            nshots (int): The number of shots.
            reviews_per_request (int): The number of reviews per request.
            num_reviews_to_process (int): The number of reviews to process.
            use_all_BC (bool): Whether to use all beer characteristcs of each review.
            wait_for_interval (bool): Whether to wait for the interval in costly hours.
            max_concurrent_batches (int): Number of batches to process in parallel.
            submit_delay_sec (float): Small delay between starting each batch request to reduce burst load.
        Returns:
            pandas.DataFrame: The DataFrame containing the ABSA results.
        """

        # Lazy imports to avoid changing module-level imports and keep compatibility
        from concurrent.futures import ThreadPoolExecutor, as_completed

        i_initial_eval_index = 52618  # 0 in from begining, otherwise index of last processed element + 1
        i_final_eval_index = min(num_reviews_to_process, len(df_base)) if num_reviews_to_process is not None else len(df_base)

        prompt_zero = self.step_4_1_get_prompt_zero_shot()
        if nshots == 0:
            prompt_n_shot = prompt_zero
        else:
            prompt_n_shot = self.step_4_1_get_prompt_few_shots(prompt_zero, nshots, use_all_BC)

        print(f'Running {step_name} with model {model} and {nshots} shots (parallel: up to {max_concurrent_batches} batches) ...')
        response_columns = ['index', 'aspect', 'category', 'sentiment']
        df_response = pd.DataFrame(columns=response_columns)
        n_shot_file_name = f'{self.work_dir}/{step_name}__{nshots}shots_{model}_{"all_BC" if use_all_BC else f"{nshots}_BC"}_{reviews_per_request}rev_per_req_from_{i_initial_eval_index}.csv'

        # if n_shot_file_name exists, read it
        if os.path.exists(n_shot_file_name):
            print(f'Reading from existing {n_shot_file_name}')
            df_response = pd.read_csv(n_shot_file_name, sep=",", encoding="utf-8")
        else:
            df_response.to_csv(n_shot_file_name, index=False, header=True)

        # Build list of indices to process, skipping already processed ones
        processed_indices = set(df_response['index'].unique()) if not df_response.empty else set()
        indices_to_process = [i for i in range(i_initial_eval_index, i_final_eval_index) if i not in processed_indices]

        # Create batches of indices of size reviews_per_request
        batches = []
        current_batch = []
        for idx in indices_to_process:
            current_batch.append(idx)
            if len(current_batch) == reviews_per_request:
                batches.append(current_batch)
                current_batch = []
        if current_batch:
            batches.append(current_batch)

        if not batches:
            print('Nothing to process. All indices already completed in this range.')

        # Helper to build the reviews payload for a batch
        def build_reviews_comments(batch_indices):
            comments = ''
            for i_general in batch_indices:
                if wait_for_interval:
                    # Respect time window per review to keep previous behavior
                    self.wait_for_interval("10:00", "02:00")
                line = df_base.iloc[i_general]
                comm = line[['review_comment']].values[0]
                comm = self.clean_json_string(comm)
                comments += f'\n{{"{i_general}", "{comm}"}}'
            return comments

        # Parsing logic extracted into a helper for reuse/thread isolation
        def parse_response_to_df(response_text):
            # Remove leading/trailing whitespace/newlines
            response_local = response_text.lstrip().rstrip()

            # Normalize start and end brackets
            response_local = re.sub(r'^\s*(?:\[\s*)+', '[[', response_local)
            response_local = re.sub(r'(?:\s*\])+\s*$', ']]', response_local)

            # Fixes for model hallucinations when processing multiple reviews
            patterns = [
                r'\s*[\r\n]*\]\s*[\r\n]*\[\s*[\r\n]*',                 # "] ["
                r'\s*[\r\n]*\]\s*[\r\n]*\]\s*[\r\n]*\[\s*[\r\n]*',  # "]] ["
                r'\s*[\r\n]*\]\s*[\r\n]*\[\s*[\r\n]*\[\s*[\r\n]*',  # "] [["
                r'\s*[\r\n]*\]\s*[\r\n]*\]\s*[\r\n]*\[\s*[\r\n]*\[\s*[\r\n]*', # "]] [["
                r'\s*[\r\n]*\]\s*[\r\n]*\]\s*[\r\n]*,\s*[\r\n]*\[\s*[\r\n]*\[\s*[\r\n]*', # "]], [["
                r'\s*[\r\n]*\]\s*[\r\n]*\]\s*[\r\n]*,\s*[\r\n]*\[\s*[\r\n]*', # "]], ["
                r'\s*[\r\n]*\]\s*[\r\n]*,\s*[\r\n]*\[\s*[\r\n]*\[\s*[\r\n]*', # "], [["
            ]
            for pat in patterns:
                response_local = re.sub(pat, '],[', response_local)

            # Fix for fenced code blocks
            response_local = response_local.replace('```json', '').replace('```', '')

            data_parsed = ast.literal_eval(response_local)

            def _is_row(x):
                return isinstance(x, (list, tuple)) and len(x) == 4

            def _is_group(x):
                return isinstance(x, (list, tuple)) and len(x) > 0 and all(_is_row(y) for y in x)

            rows_flat = []

            if isinstance(data_parsed, (list, tuple)):
                if len(data_parsed) == 0:
                    rows_flat = []
                elif all(_is_group(x) for x in data_parsed):
                    for g in data_parsed:
                        rows_flat.extend(g)
                elif all(_is_row(x) for x in data_parsed):
                    grouped = []
                    current_idx = None
                    current_group = []
                    for r in data_parsed:
                        idx = r[0]
                        if current_idx is None or idx == current_idx:
                            if current_idx is None:
                                current_idx = idx
                            current_group.append(r)
                        else:
                            grouped.append(current_group)
                            current_group = [r]
                            current_idx = idx
                    if current_group:
                        grouped.append(current_group)
                    for g in grouped:
                        rows_flat.extend(g)
                else:
                    def _collect(el, out):
                        if _is_row(el):
                            out.append(el)
                        elif isinstance(el, (list, tuple)):
                            for zz in el:
                                _collect(zz, out)
                    _collect(data_parsed, rows_flat)
            else:
                rows_flat = []

            return pd.DataFrame(rows_flat, columns=response_columns)

        error_count = 0

        # Worker that runs a single batch
        def process_batch(batch_indices):
            try:
                reviews_comments = build_reviews_comments(batch_indices)
                prompt_ai = Prompt_AI(model, f'{prompt_n_shot} {reviews_comments} ')
                response, finish_reason = prompt_ai.get_completion()
                if finish_reason != 'stop':
                    print(f'Finish reason not expected: {finish_reason}')
                    return None, 1, batch_indices

                df_new_local = parse_response_to_df(response)
                return df_new_local, 0, batch_indices
            except Exception as e:
                print(f'\n\nException while processing batch {batch_indices}: {e}')
                return None, 1, batch_indices

        # Submit batches in parallel with a small delay between submissions
        futures = []
        with ThreadPoolExecutor(max_workers=max_concurrent_batches) as executor:
            for batch in batches:
                futures.append(executor.submit(process_batch, batch))
                # small delay between requests to avoid burst
                if submit_delay_sec and submit_delay_sec > 0:
                    time_module.sleep(submit_delay_sec)

            # Collect results as they complete and write sequentially to the CSV
            for fut in as_completed(futures):
                df_new, inc_err, batch_indices = fut.result()
                error_count += inc_err
                if df_new is None:
                    continue

                # Append to in-memory df and persist incrementally
                df_response = pd.concat([df_response, df_new], ignore_index=True)
                df_new.to_csv(n_shot_file_name, mode='a', index=False, header=False)

                expected_count = len(batch_indices)
                if len(df_new) < expected_count and len(batch_indices) == reviews_per_request:
                    print(f'WARNING: Not all reviews were processed for batch starting at {batch_indices[0]}: expected {expected_count}, got {len(df_new)}')

        print(f'TOTAL Error count: {error_count}')
        # finally, sort to check responses and save all the results
        # avoids error when index is not numeric - allucination
        df_response['index'] = pd.to_numeric(df_response['index'], errors='coerce')
        df_response = df_response.sort_values(by=['index', 'aspect'])
        df_response.to_csv(n_shot_file_name, index=False)
        
        self.inp_out_df = df_response.copy()

        return df_response, n_shot_file_name

    def run(self):
        """
        This function runs Step 4 of the Aspect-Based Sentiment Analysis of Beer Characteristics.
        It reads the step_3_reviews_main.csv (Main Base) containing the reviews for the previous step, creates the prompts and then
        test models and nshots by testing different prompts. Finally, runs the best prompt in the entire Base (Main Base)

        Args:
                self (object): The object instance that contains the data.

        Returns:
        """
        
        print(f'\n\nRunning Step 4\n================================')
        file = f'{self.work_dir}/step_3_reviews_main.csv'
        self.read_inp_out_csv(file)
        
        # Creates the Base Prompts Creation: for creating one and few shot prompts based on step_3_reviews_main.csv
        df_selecao_prompts = self.run_step_4_1_create_base_prompts()
        print(df_selecao_prompts.describe())
        
        # Creates the Base Prompts Validation: used to test models and nshots
        df_reviews_sample = self.run_step_4_1_create_base_reviews_sample(df_selecao_prompts)
        print(df_reviews_sample.describe())
        
        # do ABSA in Base Reviews Sample for n shots and models, to select the best combination
        self.run_step_4_2_ABSA_model_shots_evaluation(df_reviews_sample)
       
        # do ABSA for real with the best combination of models and shots
        self.run_step_4_3_evaluate_main_base()
        self.inp_out_df.to_csv(f'{self.work_dir}/step_4_ABSA_main.csv', index=False)
