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
from pprint import pprint

# Set locale dates in spanish
locale.setlocale(locale.LC_ALL, "es_ES")


def gen_data():
    fk = Faker('es_CO')
    profile = fk.simple_profile(sex=random.choice(['M', 'F']))
    npays = fk.pyint(min_value=3, max_value=12)
    initial_date = fk.past_datetime() - timedelta(days=30 * npays)

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

    total = fk.pyfloat(min_value=100000, max_value=10000000, right_digits=2)
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

    return {'debtor': debtor, 'debt': debt}


class ConversationFactory:
    def __init__(self, input_csv, system_prompt, sub_entry=None):
        self.df = pd.read_csv(input_csv)
        self.system_prompt = system_prompt
        self.sub_entry = sub_entry

    def get_unique_conversations(self):
        return self.df.id.unique()

    def generate_conversation(self, c, cdata):
        sub = self.sub_entry or {
            'full_name': cdata['debtor']['full_name'],
            'address': cdata['debtor']['address'],
            'phone_number': cdata['debtor']['phone_number'],
            'email': cdata['debtor']['email'],
            'identification': cdata['debtor']['identification'],
            'days': cdata['debt']['ndays'],
            'amount': cdata['debt']['amount'],
            'today_date': cdata['debt']['today_date'],
            'tomorrow_date': cdata['debt']['tomorrow_date'],
            'title': cdata['debtor']['title'],
            'pronoun': cdata['debtor']['pronoun'],
            'gender_confirm': cdata['debtor']['confirmation'],
            'known': cdata['debtor']['known'],
        }

        conversation = {
            "messages": [{"role": "system", "content": Template(self.system_prompt).substitute(sub)}]
        }

        dfc = self.df[self.df.id == c].reset_index(drop=True)

        for i in range(0, len(dfc.index), 2):
            conversation['messages'].append(
                {"role": "user", "content": Template(dfc.loc[i].text).substitute(sub)}
            )
            conversation['messages'].append(
                {"role": "assistant", "content": Template(dfc.loc[i + 1].text).substitute(sub)}
            )

        return conversation

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

    def generate_training_data(self):
        conversations = []
        convs = self.get_unique_conversations()

        for c in convs:
            cdata = gen_data()
            conversation = self.generate_conversation(c, cdata)
            variations = self.get_variations(conversation)
            conversations.extend(variations)

        return conversations


if __name__ == '__main__':
    input_csv = 'conversations_azteca.csv'
    system_prompt3 = """
    Eres Raul de Banco Azteca.
    Tu objetivo es informar sobre el estado de la obligación financiera y realizar un acuerdo del pago con el deudor.
    """

    factory = ConversationFactory(input_csv, system_prompt3)
    conversations = factory.generate_training_data()

    print(f"Total training conversations: {len(conversations)}")

    with open('azteca_train_template.jsonl', 'w') as f:
        for item in conversations:
            f.write(json.dumps(item) + '\n')

    print(f"Conversation created successfully: {len(conversations)}")