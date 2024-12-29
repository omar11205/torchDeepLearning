from typing import Optional, Dict, Any, List
import random
import json
import locale
import itertools

import numpy as np
import pandas as pd
import re

from datetime import datetime, timedelta
from faker import Faker
from copy import deepcopy
from string import Template

from twisted.words.im.basechat import Conversation

# Set locale dates in spanish
locale.setlocale(locale.LC_ALL, "es_ES")
random.seed(0)


class ConversationDataGenerator:
    def __init__(self, bank_name: str, agent_name: str, additional_data: Optional[Dict[str, Any]] = None):
        self.bank_name = bank_name
        self.agent_name = agent_name
        self.additional_data = additional_data

    def gen_data(self) -> Dict[str, Any]:
        fk = Faker('es_CO')
        profile = fk.simple_profile(sex=random.choice(['M', 'F']))
        npays = fk.pyint(min_value=3, max_value=12)
        initial_date = fk.past_datetime() - timedelta(days=30 * npays)

        cobranzas = {
            'bank_name': self.bank_name,
            'agent': self.agent_name,
            'contact_number': 4775006675
        }

        gendered_terms = {
            'M': {
                'title': 'Señor',
                'pronoun': 'el',
                'confirmation': 'él',
                'known': 'lo'
            },
            'F': {
                'title': 'Señora',
                'pronoun': 'la',
                'confirmation': 'ella',
                'known': 'la'
            }
        }
        gender = profile['sex']
        debtor_terms = gendered_terms[gender]

        debtor = {
            'full_name': profile['name'],
            'first_name': profile['name'].split(' ', 1)[0],
            'gender': gender,
            'title': debtor_terms['title'],
            'pronoun': debtor_terms['pronoun'],
            'confirmation': debtor_terms['confirmation'],
            'known': debtor_terms['known'],
            'identification': str(fk.pyint(min_value=10000000, max_value=99999999)),
            'address': profile['address'],
            'email': profile['mail'],
            'phone_number': fk.phone_number()
        }

        total = fk.pyfloat(min_value=10000, max_value=1000000, right_digits=2)
        due_date = initial_date + timedelta(days=30 * npays)
        date_today = datetime.now()

        debt = {
            'status': 'activo',
            'amount': total,
            'npays': npays,
            'start_date': initial_date.isoformat(),
            'due_date': due_date.isoformat(),
            'ndays': (date_today - due_date).days,
            'outstanding_balance': round(total / npays, 2),
            'today_date': date_today.strftime("%d de %B de %Y"),
            'tomorrow_date': (date_today + timedelta(1)).strftime("%d de %B de %Y"),
            'system_date_time': date_today.strftime("%A %Y-%m-%d %I:%M %p")
        }

        if self.additional_data is not None:
            self.additional_data = deepcopy(self.additional_data)
            return {'debtor': debtor, 'debt': debt, 'cobranzas': cobranzas, 'aditional': self.additional_data}
        else:
            return {'debtor': debtor, 'debt': debt, 'cobranzas': cobranzas}


class ConversationFactory:
    def __init__(self,
                 input_csv: str,
                 system_prompt: str,
                 gen_data_instance: ConversationDataGenerator,
                 model_chat_completion: Optional[Dict[str, Any]] = None,
                 sub_entry_function: Optional[Any] = None,
                 create_random_dataset: bool = False
                 ):

        self.df = pd.read_csv(input_csv)
        self.system_prompt = system_prompt
        self.sub_entry = sub_entry_function
        self.create_random_dataset = create_random_dataset
        self.gen_data = gen_data_instance
        self.m = model_chat_completion or {  # the default chat completion is for OpenAI Models
                'messages': 'messages',
                'role': 'role',
                'content': 'content',
                'system': 'system',
                'user': 'user',
                'assistant': 'assistant'
        }

    def get_unique_conversations(self) -> np.ndarray:
        return self.df.id.unique()

    @staticmethod
    def generate_basic_sub_entry(cdata: Dict[str, Any]) -> Dict[str, Any]:
        sub = {
            'bank_name': cdata['cobranzas']['bank_name'],
            'agent': cdata['cobranzas']['agent'],
            'contact_number': cdata['cobranzas']['contact_number'],
            'full_name': cdata['debtor']['full_name'],
            'address': cdata['debtor']['address'],
            'phone_number': cdata['debtor']['phone_number'],
            'email': cdata['debtor']['email'],
            'identification': cdata['debtor']['identification'],
            'days': cdata['debt']['ndays'],
            'plural_ndays': "días" if cdata['debt']['ndays'] > 1 else "día",
            'amount': cdata['debt']['amount'],
            'today_date': cdata['debt']['today_date'],
            'tomorrow_date': cdata['debt']['tomorrow_date'],
            'title': cdata['debtor']['title'],
            'pronoun': cdata['debtor']['pronoun'],
            'gender_confirm': cdata['debtor']['confirmation'],
            'known': cdata['debtor']['known'],
            'system_date_time': cdata['debt']['system_date_time']
        }

        return sub

    def generate_conversation(self, c: int, cdata: Dict[str, Any]):
        if self.sub_entry is not None:
            sub = self.sub_entry(cdata)
        else:
            sub = self.generate_basic_sub_entry(cdata)

        conversation = {
            f"{self.m['messages']}": [{f"{self.m['role']}": f"{self.m['system']}",
                                       f"{self.m['content']}": Template(self.system_prompt).substitute(sub)}]
        }

        dfc = self.df[self.df.id == c].reset_index(drop=True)

        def fill_sub_template(conv, dfc_i, i, user_cont):
            conv[f"{self.m['messages']}"].append(
                {f"{self.m['role']}": f"{self.m['user']}", f"{self.m['content']}": user_cont}
            )
            conv[f"{self.m['messages']}"].append(
                {f"{self.m['role']}": f"{self.m['assistant']}",
                 f"{self.m['content']}": Template(dfc_i.loc[i + 1].text).substitute(sub)}
            )

        if self.create_random_dataset and self.sub_entry is None:
            for i in range(0, len(dfc.index), 2):
                user_content = Template(dfc.loc[i].text).substitute(sub)
                user_content = self.randomize_user_response(user_content)
                fill_sub_template(conversation, dfc, i, user_content)

        elif self.create_random_dataset and self.sub_entry:
            for i in range(0, len(dfc.index), 2):
                user_content = Template(Template(dfc.loc[i].text).substitute(sub)).substitute(sub)
                user_content = self.randomize_user_response(user_content)
                fill_sub_template(conversation, dfc, i, user_content)

        elif self.create_random_dataset is False and self.sub_entry is None:
            for i in range(0, len(dfc.index), 2):
                user_content = Template(dfc.loc[i].text).substitute(sub)
                fill_sub_template(conversation, dfc, i, user_content)

        elif self.create_random_dataset is False and self.sub_entry:
            for i in range(0, len(dfc.index), 2):
                user_content = Template(Template(dfc.loc[i].text).substitute(sub)).substitute(sub)
                fill_sub_template(conversation, dfc, i, user_content)

        return conversation

    def gemini_15_conversation(self, c: int, cdata: Dict[str, Any]):
        if self.sub_entry is not None:
            sub = self.sub_entry(cdata)
        else:
            sub = self.generate_basic_sub_entry(cdata)

        dfc = self.df[self.df.id == c].reset_index(drop=True)

        conversation = {
            "systemInstruction": {
                "role": "system",
                "parts": [
                    {
                        "text": Template(self.system_prompt).substitute(sub)
                    }
                ]
            },
            "contents": []
        }

        def fill_sub_template(conv, dfc_i, i, user_cont):
            conv["contents"].append(
                {"role": "user", "parts": [
                    {
                        "text": user_cont
                    }
                ]}
            )

            conv["contents"].append(
                {"role": "model", "parts": [
                    {
                        "text": Template(dfc_i.loc[i + 1].text).substitute(sub)
                    }
                ]}
            )

        if self.create_random_dataset and self.sub_entry is None:
            for i in range(0, len(dfc.index), 2):
                user_content = Template(dfc.loc[i].text).substitute(sub)
                user_content = self.randomize_user_response(user_content)
                fill_sub_template(conversation, dfc, i, user_content)

        elif self.create_random_dataset and self.sub_entry:
            for i in range(0, len(dfc.index), 2):
                user_content = Template(Template(dfc.loc[i].text).substitute(sub)).substitute(sub)
                user_content = self.randomize_user_response(user_content)
                fill_sub_template(conversation, dfc, i, user_content)

        elif self.create_random_dataset is False and self.sub_entry is None:
            for i in range(0, len(dfc.index), 2):
                user_content = Template(dfc.loc[i].text).substitute(sub)
                fill_sub_template(conversation, dfc, i, user_content)

        elif self.create_random_dataset is False and self.sub_entry:
            for i in range(0, len(dfc.index), 2):
                user_content = Template(Template(dfc.loc[i].text).substitute(sub)).substitute(sub)
                fill_sub_template(conversation, dfc, i, user_content)

        return conversation


    @staticmethod
    def randomize_user_response(content: str) -> str:
        """
        Replace options in brackets [option1; option2; ...] with one random choice.
        """
        def replace_match(match):
            options = [x.strip() for x in match.group(1).split(';')]
            return random.choice(options)

        return re.sub(r'\[(.*?)\]', replace_match, content)

    @staticmethod
    def get_variations(conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate variations of a conversation by replacing placeholders in user messages
        with all possible combinations of alternatives.

        Args:
            conversation (Dict[str, Any]): A dictionary representing a conversation. It is
                                           expected to have a 'messages' key containing a list
                                           of dictionaries. Each dictionary should include
                                           'role' and 'content' keys.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a variation of the
                                  original conversation. If no placeholders are found, the
                                  original conversation is returned in a single-item list.
        """

        variations: List[Dict[str, Any]] = []
        subs: List[Dict[str, Any]] = []

        for entry in conversation['messages']:
            if entry['role'] == 'user':
                x = re.search(r'\[(.*?)\]', entry['content'])
                if x is not None:
                    alts: List[str] = [y.strip() for y in x.group(1).split(';')]
                    subs.append({'entry': entry, 'alts': alts})

        al: List[List[str]] = [sub['alts'] for sub in subs]
        pr: List[tuple] = list(itertools.product(*al))

        for p in pr:
            for idx, sub in enumerate(subs):
                sub['entry']['content'] = p[idx]
            cc: Dict[str, Any] = deepcopy(conversation)
            variations.append(cc)

        return variations if variations else [conversation]

    @staticmethod
    def select_random_user_entries(in_list: List[str], n_choices: int) -> List[str]:
        if len(in_list) > n_choices:
            return random.sample(in_list, n_choices)
        else:
            return in_list

    @staticmethod
    def get_gemini_variations_v1(conversation: Dict[str, Any], n_variations: int) -> List[Dict[str, Any]]:
        """
        Generate variations of a conversation in Gemini 1.5 format by replacing placeholders
        in user messages with all possible combinations of alternatives.
        :param conversation: A dictionary representing a Gemini 1.5 conversation.
                                       It is expected to have a 'contents' key containing a
                                       list of dictionaries. Each dictionary should include
                                       'role' and 'parts' keys.
        :param n_variations: Integer number of máximum user entries to select randomly.
        :returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a variation of the
                                  original conversation. If no placeholders are found, the
                                  original conversation is returned in a single-item list.
        """
        variations: List[Dict[str, Any]] = []
        subs: List[Dict[str, Any]] = []

        # Step 1: Identify placeholders in 'contents' for user messages
        for entry in conversation['contents']:
            if entry['role'] == 'user':
                x = re.search(r'\[(.*?)\]', entry['parts'][0]['text'])
                if x is not None:
                    alts: List[str] = [y.strip() for y in x.group(1).split(';')]
                    if n_variations >= 2:
                        alts: List[str] = ConversationFactory.select_random_user_entries(alts, n_variations)
                    subs.append({'entry': entry, 'alts': alts})

        # Step 2: Generate Cartesian product of all alternatives
        al: List[List[str]] = [sub['alts'] for sub in subs]
        pr: List[tuple] = list(itertools.product(*al))

        # Step 3: Replace placeholders and create variations
        for p in pr:
            for idx, sub in enumerate(subs):
                sub['entry']['parts'][0]['text'] = p[idx]
            cc: Dict[str, Any] = deepcopy(conversation)
            variations.append(cc)

        return variations if variations else [conversation]

    def generate_fine_tuning_dataset(self, dataset_size: int = 1) -> List[Dict[str, Any]]:
        conversations = []
        convs = self.get_unique_conversations()

        if self.create_random_dataset:
            for _ in range(dataset_size):
                cdata = self.gen_data.gen_data()
                for c in convs:
                    conversation = self.generate_conversation(c, cdata)
                    conversations.append(conversation)

            return conversations

        else:
            for c in convs:
                cdata = self.gen_data.gen_data()
                conversation = self.generate_conversation(c, cdata)
                variations = self.get_variations(conversation)
                conversations.extend(variations)

            return conversations

    def generate_gemini_fine_tuning_dataset(self, dataset_size: int = 1, maximum_variation_control: int = 0):
        """
        Returns the fine-tuning dataset using the Gemini 1.5 format
        :param dataset_size: when create_random_dataset = True, create a number of random conversations equal to
                             dataset_size.
        :param maximum_variation_control: controls the variations when create_random_dataset = False, selecting a maximum
                                          of user entries from the total user entries.
                                          This parameter affect all the users entries.
                                          By default there are not any control on the variations (0).
                                          If you want to control the variations use an integer greater or equal than 2.

        :return: fine-tuning dataset List[Dict[str, Any]]
        """
        conversations = []
        convs = self.get_unique_conversations()

        if self.create_random_dataset:
            for _ in range(dataset_size):
                cdata = self.gen_data.gen_data()
                for c in convs:
                    conversation = self.gemini_15_conversation(c, cdata)
                    conversations.append(conversation)

            return conversations
        else:
            for c in convs:
                cdata = self.gen_data.gen_data()
                conversation = self.gemini_15_conversation(c, cdata)
                variations = self.get_gemini_variations_v1(conversation, maximum_variation_control)
                conversations.extend(variations)
            return conversations

    @staticmethod
    def generate_jsonl(conversations_list: List[Dict[str, Any]],
                       file_name: str = "fine_tuning_dataset.jsonl",
                       utf8_encoding: bool = False
                       ):

        print(f"Total conversations: {len(conversations_list)}")

        if utf8_encoding is False:
            with open(file_name, 'w') as f:
                for items in conversations_list:
                    f.write(json.dumps(items) + '\n')
        else:
            with open(file_name, 'w', encoding="utf-8") as f:
                for items in conversations_list:
                    f.write(json.dumps(items, ensure_ascii=False) + '\n')

        print(f"Conversation jsonl created successfully: {len(conversations_list)} conversations in {file_name}")




if __name__ == '__main__':
    # generate a variational dataset
    template_csv = 'chat_flux_template_azteca_test.csv'

    system_prompt_default = (
            "Tu nombre es $agent y trabajas para Cumplir SAS. " 
            "Tu tarea es comunicarte con los clientes con alta empatía y comprensión. "
            "Nombre Banco: $bank_name. "
            "Nombre Cliente: $full_name. " 
            "Monto Adeudado: $amount pesos. "
            "Fecha y hora de hoy: $system_date_time. "
            "Días de atraso en el pago: $days. " 
            "Fecha de pago máxima: $tomorrow_date. "
            "Número de contacto de Banco Azteca: $contact_number"
    )

    conversation_data_generator = ConversationDataGenerator(
        bank_name="Banco Azteca",
        agent_name="Raúl"
    )

    factory = ConversationFactory(
        input_csv=template_csv,
        system_prompt=system_prompt_default,
        gen_data_instance=conversation_data_generator,
        create_random_dataset=False,  # do not permutate the user entries
    )

    conversations = factory.generate_fine_tuning_dataset()
    print(f"Total training conversations: {len(conversations)}")

    factory.generate_jsonl(conversations, file_name="train_azteca_beta.jsonl")



