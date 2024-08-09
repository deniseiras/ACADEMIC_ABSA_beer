"""
Step 4: Aspect-Based Sentiment Analysis of Beer Characteristics (CC)

:author: Denis Eiras

Functions:
    - 
"""

import pandas as pd
import ast
from step import Step
from src.openai_api import get_completion

class Step_4(Step):


    def __init__(self) -> None:
        super().__init__()


    def run(self):
        """
        This function runs Step 4 of the Aspect-Based Sentiment Analysis of Beer Characteristics.
        It reads the csv file containing the reviews for the previous step, and then creates the 
        "Base Prompts" and the prompts for the Aspect-Based Sentiment Analysis of Beer Characteristics.

        Args:
                self (object): The object instance that contains the data.

        Returns:
        """
        
        print(f'\n\nRunning Step 4\n================================')
        file = f'{self.work_dir}/step_3.csv'
        self.read_csv(file)
        
        # reviews for creating one and few shot prompts
        reviews_for_prompts_df = self.run_step_4_1_reviews_for_prompts()
        print(reviews_for_prompts_df.describe())
        
        # create Base Prompts
        base_prompts_df = self.run_step_4_1_base_prompts(reviews_for_prompts_df)
        print(base_prompts_df.describe())
        
        # do ABSA in Base Prompts for n shots and models, to select the best combination
        self.run_step_4_2_ABSA(base_prompts_df)
        
    def run_step_4_1_base_prompts(self, reviews_for_prompts_df: pd.DataFrame):
        """
        This function selects reviews for Prompt ABSA based on certain criteria.
        Creates the "Base Prompts" base, for using in prompts one and few shots.
        Parameters:
            self (object): The object instance that contains the data.
        Returns:
            pandas.DataFrame: A DataFrame containing the selected reviews. The DataFrame is sorted by beer style, 
            review general rate, and review number of reviews.
        """            
       
        # Base Prompts creation
        print(f'Step 4.1 - Base Prompts creation')
        print(f'- Initial line count: {len(self.df)}')
        df = self.df

        # remove registers of df containing reviews_for_prompts_df registers
        df = df[~df.isin(reviews_for_prompts_df)]
        print(f'- Removing registers from reviews_for_prompts_df - Parcial line count (Base Prompts): {len(df)}')
        
        # review_comment size >= 75% of the greatest sizes 
        greatest_review_comment_size_threshold = df['review_comment_size'].quantile(0.75) 
        df = df[df['review_comment_size'] >= greatest_review_comment_size_threshold]
        print(f'- Select greatest review comment size - Parcial line count (Base Prompts): {len(df)}')
        
        # Best and worst reviews
        df = df[(df['review_general_rate'] >= 4.0) | (df['review_general_rate'] <= 2.0)]
        print(f'- Select best and worst reviews - Parcial line count (Base Prompts): {len(df)}')
        
        # percentil 1% reviwers with most reviews “review_num_reviews” and inexperient
        greatest_reviewers_threshold = df['review_num_reviews'].quantile(0.99) 
        df = df[ (df['review_num_reviews'] >= greatest_reviewers_threshold) | (df['review_num_reviews'] == 1)]
        print(f'- Select experients and inexperients - Parcial line count (Base Prompts): {len(df)}')

        # select only one register per review_user column on df
        df = df.groupby('review_user').head(1)
        print(f'- Select only one register per review_user - Parcial line count (Base Prompts): {len(df)}')
        
        # select only one register per beer_style column on df
        df = df.groupby('beer_style').head(1)
        print(f'- Select only one register per beer_style - Parcial line count (Base Prompts): {len(df)}')
        
        
        df = df.sort_values(by=['beer_style', 'review_general_rate', 'review_num_reviews'])
        print(f'- Final line count (Base Prompts): {len(df)}')
        
        df.to_csv(f'{self.work_dir}/step_4_1__base_prompts.csv', index=False)
        return df


    def run_step_4_1_reviews_for_prompts(self):
        """
        This function selects reviews for Prompt ABSA based on certain criteria.
        Parameters:
                self (object): The object instance that contains the data.

        Returns:
                pandas.DataFrame: A DataFrame containing the selected reviews. The DataFrame is sorted by beer style, 
                review general rate, and review number of reviews.
        """

        print(f'Step 4.1 - Selections of reviews for Prompts ABSA')
        styles_for_prompt = ['India Pale Ale (IPA)', 'German Weizen', 'Porter', 'Witbier']
        df = self.df[ self.df['beer_style'].isin(styles_for_prompt) & 
                                     ((self.df['review_general_rate'] >= 4) | (self.df['review_general_rate'] <= 2)) & 
                                     # TODO check 368
                                     ((self.df['review_num_reviews'] >= 368) | (self.df['review_num_reviews'] == 1))]
        df = df.sort_values(by=['beer_style', 'review_general_rate', 'review_num_reviews'])
        
        df.to_csv(f'{self.work_dir}/step_4_1__reviews_for_prompt.csv')
        
        return df


    def step_4_1_get_prompt_zero_shot(self):
            
        print(f'Step 4.1 - "Prompt ABSA zero-shot" creation')
        prompt_sys = """ 
        "Você é um extrator de aspectos de cerveja. Do texto, extraia os ‘aspectos’ e a ‘categoria’ relacionados aos aspectos da cerveja. As categorias devem estar
dentre os valores: ‘visual’, ‘aroma’, ‘sabor’, ‘amargor’, ‘álcool’ e ‘sensação na boca’. Extraia o ‘sentimento’ dentre os valores ‘muito negativo’, ‘negativo’, ‘neutro’,
‘positivo’ ou ‘muito positivo’ para cada par aspecto/categoria. 
 
        As avaliações estão na lista compreendida entre colchetes, onde cada item contém "index", que registra o índice da avaliação e \
    "review_comment" o texto a ser avaliado. 

        Não faça comentários, apenas gere a saída no formato: [['index', 'aspecto', 'categoria', 'sentimento' ]].
"""
        return prompt_sys

    def step_4_1_get_prompt_one_shot(self, prompt_zero_shot: str):

        print(f'Step 4.1 - "Prompt ABSA one-shot" creation')
        prompt_one_shot = prompt_zero_shot + """
        Um exemplo abaixo entre aspas e o resultado esperado em seguida.
        "Rótulo agradável, em garrafa âmbar bojuda. Tampa sem rótulo, dando um aspecto desleixado à cerveja. As cervejas bastante lupuladas sempre têm uma agradável
antecipação do aroma logo quando se abre a garrafa. Essa não tinha: mau presságio... Cor âmbar, translúcida,excelente sensação visual ao ser servida, particularmente
pela intensa formação de espuma, que é persistente. Aroma herbáceo, suave demais,muito aquém para uma cerveja que carrega no lúpulo aromático, inclusive tendo sido 
feito dry hopping com Cascade, um lúpulo essencialmente aromático . Se a intenção era fazer uma autêntica American IPA, vê-se aqui mais pra uma do velho continente, 
inglesa, precisamente. No sabor,perde-se completamente: tem um amargor intenso mas adstringente, incomodativo, que parece mesmo arranhar a língua e que perdura no 
aftertaste.Baixíssima drinkability. Vê-se muito malte, particularmente no aftertaste, quando se mantém um retrogosto de mel e pão. Não vi qualquer off-flavor no exemplar
que degustei. Pergunto: é uma IPA ou é uma American Pale Ale lupulada em excesso (amargor excessivo, por ser incomodativo...)?"
[['Cor âmbar', 'visual', 'neutro'],
 ['translúcida', 'visual', 'neutro'],
 ['excelente sensação visual', 'visual', 'muito positivo'},
 ['espuma persistente', 'visual', 'muito positivo'},
 ['herbáceo suave demais', 'aroma', 'negativo'},
 ['adstringente', 'amargor', 'negativo'},
 ['muito malte', 'sabor', 'neutro'},
 ['retrogosto de mel', 'sabor', 'neutro'},
 ['retrogosto de pão', 'sabor', 'neutro'},
 ['sem off-flavor', 'aroma', 'positivo'},
 ['lupulada em excesso', 'sabor', 'negativo'},
 ['amargor excessivo', 'amargor', 'negativo'} ]
"""
        return prompt_one_shot

    def step_4_1_get_prompt_few_shots(self, prompt_zero_shot: str):
            
                    
        # beer_style review_user review_num_reviews review_general_rate review_comment
        #
        # ***** Wibier
        #
        # - experienced - low rate
        # Bruno Sicchieri	531	1.1
        print(f'Step 4.1 - "Prompt ABSA few-shots" creation')
        style1_exp_lowrate = """De coloração amarelada, turva. Espuma de difícil formação, altamente efervescente e sem duração. Bom aroma \
trazendo notas cítricas de laranja e semente de coentro. Na boca, início e final amargos e efervescentes, quanto ao sabor... horrível... \
agitei para capturar um pouco do fermento sedimentando no fundo e creio que foi meu erro... é difícil descrever, exceto a sensação de estar \
estragada... sabor de giz e terra. Carbonatação baixa. Corpo médio. Uma terrível [BJCP2015] 24A: Witbier. Poupe suas papilas gustativas... \
ou experimente por sua própria conta e risco."""
        #
        # - experienced - high rate
        # Fabio Vieira	907	4.4	 
        style1_exp_highrate = """Temperatura de degustação: Cinco graus Celsius. Cor: Amarelo-palha medianamente turva. Creme: Média formação \
de creme branco que mantém uma fina camada persistente, deixando marcas no tumbler. Aroma: Cítrico com notas de limão, especiarias como coentro\
e pimenta, muito bom. Sabor: Maltado com cereais, frutado de limão e especiarias dominam os sentidos. O final do gole apresenta-se levemente \
amargo, levemente ácido e picante. O sabor cítrico do limão permanece por todo o gole, se prolongando no retrogosto, apresentando excelente \
drinkability e refrescância absurda! Excelente breja!!"""
        #
        # - inexperienced - low rate
        # Thiago Meireles	1	1.7	
        style1_inexp_lowrate = """A cerveja é bem docinha; minha opinião sobre ela, no entanto, é um pouco amarga. Produzida em meio à febre \
de cervejas artesanais que atingiu a burguesada do Rio, o rótulo foi metido goela abaixo do consumidor pelos principais pontos de venda de \
cervejas especiais da cidade, inclusive supermercados como o Zona Sul, em que é possível comprar boas brejas, como a Coruja, por exemplo. \
Nas prateleiras, a Niña, uma Wit Bier, ocupa mais espaço que todas as outras, inclusive as da Ambev, por isso é impossível não notá-la. Pra \
quem tem um paladar mais sensível, pode ser até boa posto que é doce, com gosto bem forte de limão - dizem no rótulo ser da variedade \
siciliano. Mas uma garrafa, que tem meros 300 ml, já é suficiente para enjoar do sabor. O dulçor esconde um peso, uma sensação de estufamento \
que vem pouco tempo após o consumo, defeito inadmissível para uma cerveja que se propõe leve acima de qualquer outra característica. O final na \
boca e na garganta é ácido. No fim, não acho que vale o preço de 11 reais na promoção no Zona Sul - o normal é encontrá-la por 14. Se daqui a \
pouco ela estiver sendo vendida a 8 pelo menos vamos saber que não vai durar muito. Sorte aos produtores que, certamente, têm dinheiro e boas \
conexões no mundo do varejo e da mídia."""
        #
        # - inexperienced - high rate
        # Robson Grespan	1	5	
        style1_inexp_highrate = """Excelente cerveja de trigo receita tipo belga. Produzida com os ingredientes: Semente de coentro, casca de \
laranja, alfarroba, baunilha, Tamara e anis estrelado. Extremamente aromática e refrescante. O alfarroba foi inserido para ter uma espuma densa\
e aveludada. Uma excelente cerveja para agradar os iniciantes e cervejeiros. Refermentada na própria garrafa.  Alfarroba na Cerveja 65 anos \
Apesar de não ter a fama do cacau, a alfarroba já era usada pelos egípcios há mais de 5 mil anos. Por ser naturalmente doce, dispensa o uso de \
açúcar na fabricação e no consumo dos produtos. Sem falar que também não possui os estimulantes cafeína e teobromina e é rica em vitaminas e \
minerais. Na cerveja “65 anos”, produz efeito espessante, dando mais corpo e textura aveludada. Além disso, os açúcares digeridos pelas leveduras\
trazem aromas delicados e únicos. Baunilha de Madagascar na Cerveja 65 anos A baunilha é a vagem seca de uma orquídea. O perfil aromático depende\
das condições de cultivo e de preparação, mas também das variedades ou espécies utilizadas. A mais tradicional é a Baunilha Bourboun, utilizada \
nesta receita e produzida em Madagascar. A idéia de utilização dela na cerveja “65 anos” é atuar no processo de refinamento dos aromas complexos \
provenientes da levedura e reestruturação do flavor da cerveja com características únicas da baunilha. Tâmara na Cerveja 65 anos As tâmaras são \
digeridas completamente depois de um longo período, pois são ricas em açúcares complexos; esta característica é bem apreciada por aqueles que \
necessitam preservar um ritmo enérgico durante atividades físicas ou mentais, normalmente em desportos que testam a resistência ou em esportes de \
duração prolongada. No caso da cerveja, esses açúcares, por serem complexos, não serão digeridos completamente pela Levedura, gerando um sabor e \
leve dulçor bem prazeroso na cerveja."""
        #
        #
        # ***** German Weizen
        # - experienced - low rate
        #  Jota Fanchin Queiroz	563	1.2	
        style2_exp_lowrate = """Uma weiss significativamente inferior ao padrão do estilo. E nem falo em comparação com as bávaras mas com a \
Eisenbahn por exemplo. Aparência: coloração dourada clara turva com creme de média formação e baixa persistência. Aroma: acanhado. Sabor: \
notas de banana e nada de cravo com um final doce demais. Estranho. Corpo: aguado até para pilsen que dirá weiss. Final: estranho, seco e \
curto. Conjunto: desequilibrado pelo excesso do doce e pelo descompassado do corpo e carbonatação. Drinkability baixa e refrescância \
comprometida."""
        #
        # - experienced - high rate
        # Eduardo Guimarães Insta @cervascomedu	2380	4,4	
        style2_exp_highrate = """Apresentou coloração dourada com espuma branca de média formação e longa persistência. \
No aroma temos banana, cravo, mel, floral e pão doce. Na boca as notas permanecem, complementadas por cereais, herbal sutil e toques \
picantes. Tem corpo médio, carbonatação moderada e sensação refrescante. Excelente!"""
        #
        # - inexperienced - low rate
        # deivis fontes	1	1	
        style2_inexp_lowrate = """deixei na geladeira por um dia e meio a garrafa em pé, percebi que ela nao apresenta tanto corpo caracteristicos \
das cervejas de trigo, talvez por ser uma cerveja industrial"""
        #
        # - inexperienced - high rate 
        # Marcelo Azambuja	1	4,9	
        style2_inexp_highrate = """A cor amarelada bem turva é algo que me agrada muito em uma Weiss, e a Alenda atende este quesito como poucas. \
Eu adquiri minhas amostras diretamente com o produtor. Já vieram resfriadas e eu as mantive assim até chegarem na minha geladeira. Desta \
forma, posso afirmar que mantém as características perfeitamente. O gosto forte, marcante, e a cremosidade do líquido são excelentes. \
Degustamos nossas amostras em nossa casa de praia (Capão da Canoa/RS) com um vizinho alemão que passa as férias no Brasil, e a frase do \
alemão (funcionário - diretor - da Mercedes-Benz, lá na Matriz em Affalterbach, um cara extremamente exigente e que conhece as principais\ 
cervejarias e países do mundo): "pode parabenizar este produtor, muito boa". Depois, por duas vezes, logo após ele tomar um pouco da cerveja,\
ele parava a conversa e dizia: "nossa, muito boa". Para quem conhece europeus, eles não são de muita gentileza, muito menos de falar algo \
que não seja totalmente sincero. Esta avaliação me ajudou muito a poder considerar a Alenda uma cerveja realmente acima da média."""
        #
        #
        # ***** India Pale Ale (IPA)
        # - experienced - low rate
        # Wagner Gasparetto	700	1,5	
        style3_exp_lowrate = """Cor amarela clara, com certa turbidez, de cara fugindo um pouco da expectativa do estilo. Aroma maltado com \
cítrico muito suave e paladar maltado, pouco lupulado e quase sem presença cítrica. Longe de uma IPA. Média carbonatação e boa drinkability,\
corpo leve. Desagradou...."""
        #
        # - experienced - high rate
        # Alexandre LC	571	4,7	
        style3_exp_highrate = """Pataqueparéu, não sei o que dizer sobre esta cerveja! Sorvida e provada logo em seguida a perigosa. Coloração âmbar\
alaranjada. Espuma levemente bege, com alta formação e boa duração. Apesar da tampinha ser o mesmo problema que a Perigosa, como foi bem \
linda no copo leva 5/5 em aparência. Aroma é fodástico, aparecendo com um buquê fenomenal. Percepção floral, cítrica, caramelada, de melaço\
e de chocolate cremoso (lembra muito o GALAK®). Com notas herbais e de laranja ao fundo. Um conjunto bem equilibrado e perfeito. \
Perfumadíssima. Aroma pra mim é 6/5! kkkk Sabor é inicialmente doce, doce de chocolate cremoso/branco, cacau, caramelo/toffe, logo mesclado\
com um amargor leve e um malte torrado bem sutil. Corpo denso e licoroso. Conjunto equilibrado e primoroso, no qual o doce inicial se acerta\
e abraça bem o amargor floral final. Final seco e levemente amargo. Retrogosto amargo e denso. SENSACIONAL. É uma IPA diferente, devido ao \
fato de o seu padrão puxar muito mais pro doce do que pro amargor lupulento, não compararei com as demais IPAs, pra mim entraria como uma \
Specialty Beer. Já está entre as minhas favoritas. Mais um preço abusivo da Bodebrown... quase R$7 por 100mL. Vacilo."""
        #
        # - inexperienced - low rate
        # Thiago Coelho	1	1,5	
        style3_inexp_lowrate = """Rótulo agradável, em garrafa âmbar bojuda. Tampa sem rótulo, dando um aspecto desleixado à cerveja. As cervejas bastante\
lupuladas sempre têm uma agradável antecipação do aroma logo quando se abre a garrafa. Essa não tinha: mau presságio... Cor âmbar, translúcida,\
excelente sensação visual ao ser servida, particularmente pela intensa formação de espuma, que é persistente. Aroma herbáceo, suave demais,\
muito aquém para uma cerveja que carrega no lúpulo aromático, inclusive tendo sido feito dry hopping com Cascade, um lúpulo essencialmente \
aromático . Se a intenção era fazer uma autêntica American IPA, vê-se aqui mais pra uma do velho continente, inglesa, precisamente. No sabor,\
perde-se completamente: tem um amargor intenso mas adstringente, incomodativo, que parece mesmo arranhar a língua e que perdura no aftertaste.\
Baixíssima drinkability. Vê-se muito malte, particularmente no aftertaste, quando se mantém um retrogosto de mel e pão. Não vi qualquer \
off-flavor no exemplar que degustei. Pergunto: é uma IPA ou é uma American Pale Ale lupulada em excesso (amargor excessivo, por ser \
incomodativo...)?"""
        #
        # - inexperienced - high rate
        # Odonio dos Anjos Filho	1	4,7	
        style3_inexp_highrate = """Cerveja com sabor de cerveja forte. Lúpulo e álcool presentes que dão o perfeito sabor de cerveja India \
Palle Ale. Mais fantástico ainda reconhecer uma cerveja dessa no Brasil, respeitando os processos de pureza necessários para fabricação de \
grandes cervejas. Vale tomar com comidas mais fortes e apreciar durante todo o ano. Espuma maravilhos que matém o aroma da cerveja de forma\
prolongada. Uma perfeição em termos de equilíbrio. Sensacional!"""
        
        #
        # ***** Porter
        # - experienced - low rate
        # Alexandre LC	571	1,7	
        style4_exp_lowrate = """Coloração negra opaca. Espuma bege de alta formação e pouca duração. Aroma de caramelo e açúcar mascavo. Sabor quase \
exclusivo de caramelo, com leve torrado e um dulçor muito acima da média, enjoativa demais. Praticamente uma malzbier menos doce. Totalmente \
fora do estilo. Bebi apenas um copo e deixei o resto para mulherada."""
        #
        # - experienced - high rate
        # Odimi Toge	1031	4,6	
        style4_exp_highrate = """Bebida desenvolvida em parceria com a Cachaçaria Nacional - maior varejista de cachaças do mundo, sediada em Belo \
Horizonte (MG).  Trata-se de um blend de Baltic Porter com a cachaça Legítima de Minas, na proporção de 10%.  Envelhecida por dois anos em \
barris de amburana, esta cachaça é produzida em Itaverava (MG) no Alambique Taverna de Minas.  A receita toda, criada pelo cervejeiro caseiro \
Fábio Ferreira, foi medalha de Ouro do XII Concurso da Acerva Mineira.  Aroma intenso de cachaça, passando por coco, canela, baunilha e mel. \
Toffee, melaço e ameixa seca surgem sinérgicos. Espetáculo! Líquido castanho avermelhado, permitindo certa passagem de luz. Servido, forma uma \
camada fina e efêmera de espuma bege clara. Na boca mostra corpo médio e reduzida carbonatação. A junção de cachaça e cerveja conversa bem, \
resultando em notas de coco queimado, canela, baunilha, ameixa seca e café - riscadas por leve dulçor maltado. Álcool inacreditavelmente bem \
inserido (sério, cadê esse álcool todo anunciado?) O final segue ligeiramente adocicado, com bastante cachaça e breve torrado.  "Drinkability" \
relativamente alta em vista de toda sua "periculosidade", por assim dizer.  Blend muito bem construído, com cerveja e cachaça na mais perfeita \
harmonia. Parabéns aos envolvidos! ????"""
        #
        # - inexperienced - low rate
        # FABIO NASCIMENTO	1	1,5	
        style4_inexp_lowrate = """Fiz a degustação da Zehn Bier - Porter e aqui vai o que percebi. Estou iniciando no mundo cervejeiro e estou tentando \
aprender a degustar estas ótimas cerveja. Lá vai: Aroma, achei adocicado,sabor pouco amarga, pouca espuma(Acho que fiz algo errado, pois no \
rótulo diz que a espuma é duradoura;não cremosa ou sem nenhuma cremosidade. Cerveja leve. Senti um pouco do sabor torrado mas não o de caramelo.\
Sabor que deixou amargo duradouro."""
        #
        # - inexperienced - high rate
        # Tiago Cosmai	1	4,6	
        style4_inexp_highrate = """Cerveja deliciosa, aroma e sabor de café presentes do início ao fim, sensação de estalar no meio da língua com a baixa \
gaseificação, cor forte típica das Porters com uma espuma pouco densa de cor caramelo escuro tão característica como o corpo da cerveja, para \
mim a Colorado Demoiselle é a melhor nacional."""

        prompt_few_shots = prompt_zero_shot + "Abaixo, textos de avaliações, onde cada texto está compreendido entre aspas e seu respectivo resultado esperado em seguida."
        prompt_few_shots += style1_exp_lowrate
        prompt_few_shots += style1_exp_highrate
        prompt_few_shots += style1_inexp_lowrate
        prompt_few_shots += style1_inexp_highrate
        prompt_few_shots += style2_exp_lowrate
        prompt_few_shots += style2_exp_highrate
        prompt_few_shots += style2_inexp_lowrate
        prompt_few_shots += style2_inexp_highrate
        prompt_few_shots += style3_exp_lowrate
        prompt_few_shots += style3_exp_highrate
        prompt_few_shots += style3_inexp_lowrate
        prompt_few_shots += style3_inexp_highrate
        prompt_few_shots += style4_exp_lowrate
        prompt_few_shots += style4_exp_highrate
        prompt_few_shots += style4_inexp_lowrate
        prompt_few_shots += style4_inexp_highrate

        return prompt_few_shots


    def run_step_4_2_ABSA(self, base_prompts_df):

        i_initial_eval_index = 0  # 0 in from begining, otherwise index of last processed element + 1
        i_final_eval_index = len(base_prompts_df)
        i_final_eval_index = 5
        reviews_per_request = 5  # api is limiting to 10 reviews per request, even when the token limit is not reached
        review_eval_count = 1
        reviews_comments = ''
        # df to validate the results 
        response_columns = ['index', 'aspect', 'category', 'sentiment']
        df_response = pd.DataFrame(columns=response_columns)
        
        # for model in ...
        model = 'gpt-4o-mini'    
        
        # for prompt_type in [ ''zero_shot', 'one_shot', 'few_shots']
        prompt_type = 'zero_shot'
        prompt_zero_shot = self.step_4_1_get_prompt_zero_shot()
        if prompt_type == 'zero_shot':
            prompt_n_shot = prompt_zero_shot
        elif prompt_type == 'one_shot':
            prompt_n_shot = self.step_4_1_get_prompt_one_shot(prompt_zero_shot)
        elif prompt_type == 'few_shots':
            prompt_n_shot = self.step_4_1_get_prompt_few_shots(prompt_zero_shot)
        
        print(f'Running {prompt_type} task with {model} model...')
        
        n_shot_file_name = f'{self.work_dir}/step_4_1__{prompt_type}_{model}.csv'
        df_response.to_csv(n_shot_file_name, index=False, header=True)
        
        for i_general in range(i_initial_eval_index, i_final_eval_index):
            line = base_prompts_df.iloc[i_general]
            
            comm = line[['review_comment']].values[0]
            comm = self.clean_json_string(comm)
            reviews_comments += f'\n["{i_general}", "{comm}"]'
            
            if review_eval_count == reviews_per_request or i_general == i_final_eval_index-1:
                # TODO - using prompt_sys in second argument makes the output json return without "[ ]"
                response, finish_reason = get_completion(f'{prompt_n_shot} {{ {reviews_comments} }}',model=model)
                if finish_reason != 'stop':
                    print(f'Finish reason not expected: {finish_reason}')
                    exit(-1)
                try:
                    data_list = ast.literal_eval(response)
                    df_new = pd.DataFrame(data_list, columns=response_columns)
                    df_response = pd.concat([df_response, df_new], ignore_index=True)
                    # saves sometimes to do not loose work 
                    df_new.to_csv(n_shot_file_name, mode='a', index=False, header=False)
                
                except Exception as e:
                    print(f'\n\nException:{e}')
                    print(f'\nError creating df: Check:\n {response}')

                review_eval_count = 0
                reviews_comments = ''
                # WARNING if it was processed all data - due to limitations of request size
                # or some data not processed due (empty response)
                if len(df_new) < reviews_per_request and i_general != i_final_eval_index-1:
                    print(f'WARNING: Not all reviews were processed, expected {reviews_per_request}, got {len(df_new)}')
                    print(f'Last review = {i_general}')
        
            review_eval_count += 1
        
        # finally, sort to check responses and save all the results
        df_response = df_response.sort_values(by=['index', 'category', 'aspect'])

        df_response.to_csv(n_shot_file_name, index=False)
        
    def run_step_4_3(self):
        pass

            
