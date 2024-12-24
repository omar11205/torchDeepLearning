import random
import json
import locale
import itertools
import pandas as pd
import re

from datetime import datetime, timedelta
from faker import Faker
from copy import deepcopy
from string import Template

# Set locale dates in spanish
locale.setlocale(locale.LC_ALL, "es_ES")


class ConversationDataGenerator:
    def __init__(self, bank_name, agent_name, additional_data=None):
        self.bank_name = bank_name
        self.agent_name = agent_name
        self.additional_data = additional_data

    def gen_data(self):
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
    def __init__(self, input_csv, system_prompt, gen_data_instance, sub_entry=None, use_random_variations=False):
        self.df = pd.read_csv(input_csv)
        self.system_prompt = system_prompt
        self.sub_entry = sub_entry
        self.use_random_variations = use_random_variations
        self.gen_data = gen_data_instance

    def get_unique_conversations(self):
        return self.df.id.unique()

    @staticmethod
    def generate_basic_sub_entry(cdata):
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

    def generate_conversation(self, c, cdata):
        if self.sub_entry is not None:
            sub = self.sub_entry(cdata)
        else:
            sub = self.generate_basic_sub_entry(cdata)

        conversation = {
            "messages": [{"role": "system", "content": Template(self.system_prompt).substitute(sub)}]
        }

        dfc = self.df[self.df.id == c].reset_index(drop=True)

        if self.use_random_variations and self.sub_entry is None:
            for i in range(0, len(dfc.index), 2):
                user_content = Template(dfc.loc[i].text).substitute(sub)
                user_content = self.randomize_user_response(user_content)
                conversation['messages'].append(
                    {"role": "user", "content": user_content}
                )
                conversation['messages'].append(
                    {"role": "assistant", "content": Template(dfc.loc[i + 1].text).substitute(sub)}
                )
        elif self.use_random_variations and self.sub_entry:
            for i in range(0, len(dfc.index), 2):
                user_content = Template(Template(dfc.loc[i].text).substitute(sub)).substitute(sub)
                user_content = self.randomize_user_response(user_content)
                conversation['messages'].append(
                    {"role": "user", "content": user_content}
                )
                conversation['messages'].append(
                    {"role": "assistant", "content": Template(dfc.loc[i + 1].text).substitute(sub)}
                )
        elif self.use_random_variations is False and self.sub_entry is None:
            for i in range(0, len(dfc.index), 2):
                conversation['messages'].append(
                    {"role": "user", "content": Template(dfc.loc[i].text).substitute(sub)}
                )
                conversation['messages'].append(
                    {"role": "assistant", "content": Template(dfc.loc[i + 1].text).substitute(sub)}
                )
        elif self.use_random_variations is False and self.sub_entry:
            for i in range(0, len(dfc.index), 2):
                conversation['messages'].append(
                    {"role": "user", "content": Template(Template(dfc.loc[i].text).substitute(sub)).substitute(sub)}
                )
                conversation['messages'].append(
                    {"role": "assistant", "content": Template(dfc.loc[i + 1].text).substitute(sub)}
                )

        return conversation

    @staticmethod
    def randomize_user_response(content):
        """
        Replace options in brackets [option1; option2; ...] with one random choice.
        """
        def replace_match(match):
            options = [x.strip() for x in match.group(1).split(';')]
            return random.choice(options)

        return re.sub(r'\[(.*?)\]', replace_match, content)

    @staticmethod
    def get_variations(conversation):
        variations = []
        subs = []

        for entry in conversation['messages']:
            if entry['role'] == 'user':
                x = re.search(r'\[(.*?)\]', entry['content'])
                if x is not None:
                    alts = [y.strip() for y in x.group(1).split(';')]
                    subs.append({'entry': entry, 'alts': alts})

        al = [sub['alts'] for sub in subs]
        pr = list(itertools.product(*al))

        for p in pr:
            for idx, sub in enumerate(subs):
                sub['entry']['content'] = p[idx]
            cc = deepcopy(conversation)
            variations.append(cc)

        return variations if variations else [conversation]

    def generate_fine_tuning_dataset(self, dataset_size=1):
        conversations = []
        convs = self.get_unique_conversations()

        if self.use_random_variations:
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

    @staticmethod
    def generate_jsonl(conversations_list, file_name="fine_tuning_dataset.jsonl",):
        print(f"Total conversations: {len(conversations_list)}")

        with open(file_name, 'w') as f:
            for items in conversations_list:
                f.write(json.dumps(items) + '\n')

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
        use_random_variations=False,  # permutate the user entries
    )

    conversations = factory.generate_fine_tuning_dataset()
    print(f"Total training conversations: {len(conversations)}")

    factory.generate_jsonl(conversations, file_name="train_azteca_beta.jsonl")



