import random
import json
import locale
from datetime import datetime, timedelta
from pandas import DataFrame
from string import Template
from chatbot_sim_template import ConversationFactory as CFactory


# Set locale dates in spanish
locale.setlocale(locale.LC_ALL, "es_ES")


# Manage client data: loads the content of a json file name of the client and the gender
class InputManager:
    def __init__(self, filename):
        """
        Initialize InputManager loads all the available data in the client data jsonl file
        format of the dataset: {'gender': gender, 'name': name}.
        :param filename: Path to the client data file.
        """
        self.filename = filename
        self.clients = self.load_client_names()

    def load_client_names(self):
        """
        Load client names from the file.
        :return: List of client data.
        """
        with open(self.filename, "r", encoding="utf-8") as file:
            return [json.loads(line) for line in file]

    @staticmethod
    def find_duplicates(data):
        """
        Generate a list of duplicate clients.
        :return: List of duplicate clients.
        """
        seen_names = set()
        duplicates = set()
        for client in data:
            name = client["name"]
            if name in seen_names:
                duplicates.add(name)
            else:
                seen_names.add(name)
        return duplicates

    def show_duplicates(self, data):
        """
        Display duplicate client names, if any.
        """
        duplicates = self.find_duplicates(data)
        if duplicates:
            print("Total duplicated names:", len(duplicates))
            print("Duplicate names found:", duplicates)
        else:
            print("No duplicate names found.")

    def get_sample(self, percentage=None, absolute=None):
        """
        Get a random sample from the loaded client data.
        :param percentage: Percentage of data to sample (1-100).
        :param absolute: Absolute number of data points to sample.
        :return: List of sampled client data.
        :raises ValueError: If both or neither of 'sample' and 'absolute' are provided.
        """
        if percentage is not None and absolute is not None:
            raise ValueError("Provide only one of 'sample' or 'absolute', not both.")

        if percentage is not None:
            if percentage <= 0 or percentage > 100:
                raise ValueError("Sample percentage must be between 1 and 100.")
            sample_size = max(1, int(len(self.clients) * percentage / 100))
        elif absolute is not None:
            if absolute <= 0 or absolute > len(self.clients):
                raise ValueError("Absolute sample size must be between 1 and the total number of clients.")
            sample_size = absolute
        else:
            raise ValueError("Either 'sample' or 'absolute' must be provided.")

        return random.sample(self.clients, sample_size)


class DatesGenerator:
    def __init__(self, holidays=None):
        # default holidays in Mexico
        self.holidays = holidays or [
            datetime(2024, 1, 1),  # Año Nuevo
            datetime(2024, 2, 5),  # Día de la Constitución (primer lunes de febrero)
            datetime(2024, 3, 18),  # Natalicio de Benito Juárez (tercer lunes de marzo)
            datetime(2024, 5, 1),  # Día del Trabajo
            datetime(2024, 9, 16),  # Día de la Independencia
            datetime(2024, 11, 18),  # Revolución Mexicana (tercer lunes de noviembre)
            datetime(2024, 12, 25)  # Navidad
        ]

    @staticmethod
    def is_business_day(date, holidays=None):
        holidays = holidays or []
        return date.weekday() < 5 and date not in holidays

    @staticmethod
    def date_range(start_date, end_date):
        current_date = start_date
        while current_date <= end_date:
            yield current_date
            current_date += timedelta(days=1)

    def generate_random_business_date_times(self, start_date, end_date, num_dates):
        business_days = [
            date for date in self.date_range(start_date, end_date)
            if self.is_business_day(date, self.holidays)
        ]
        if num_dates > len(business_days):
            raise ValueError("Not enough business days in the specified range.")

        random_dates = random.sample(business_days, num_dates)
        business_date_times = []
        for date in random_dates:
            random_hour = random.randint(7, 18)  # 7 AM to 6 PM inclusive
            random_minute = random.randint(0, 59)
            random_second = random.randint(0, 59)
            business_date_time = datetime.combine(date, datetime.min.time()) + timedelta(
                hours=random_hour, minutes=random_minute, seconds=random_second
            )
            business_date_times.append(business_date_time)

        return business_date_times


class RandomEntryGenerator:
    def __init__(self, mode, ag_names, cli_names, bank, date_config=None):
        self.mode = mode
        self.ag_names = ag_names
        self.cli_names = cli_names
        self.bank = bank
        self.date_config = date_config

    def generate_random_entries(self, start_date, end_date):
        """
        Generate a dataset containing random entries, as many as elements are in 'ag_names'.
        :param start_date:
        :param end_date:
        :return: list of dictionaries containing random entries.
        """

        r_entries = []

        for client_name in self.cli_names:

            if self.mode == "current_day":  # use current day for testing
                # current date
                date_manager = datetime.now()
            elif self.mode == "random_day":  # use random date for training
                # random business days. end_date must be less the current date
                date_manager = self.date_config.generate_random_business_date_times(start_date, end_date, 1)[0]
            else:
                date_manager = datetime.now()

            agent_name = random.choice(self.ag_names)
            days_late = random.randint(1, 60)
            amount_pesos = random.randint(1000, 50000)

            # format dates and times for conversation content
            current_date = date_manager.strftime("%d de %B de %Y")
            tomorrow_date = (date_manager + timedelta(days=1)).strftime("%d de %B de %Y")
            am_pm = "AM" if date_manager.hour < 12 else "PM"
            ask_for_payment_day = (date_manager + timedelta(days=1)).strftime("%A")

            # format dates and times for system content
            system_current_date_time = date_manager.strftime(f"%A %Y-%m-%d %I:%M {am_pm}")
            system_tomorrow_date = (date_manager + timedelta(days=1)).strftime("%A %Y-%m-%d")

            # create the entry
            r_entries.append({
                "ask_for_payment_day": ask_for_payment_day,
                "system_current_date_time": system_current_date_time,
                "system_current_datetime_object": date_manager,
                "system_tomorrow_date": system_tomorrow_date,
                "current_date": current_date,
                "tomorrow_date": tomorrow_date,
                "name_of_the_agent": agent_name,
                "days_late": days_late,
                "amount_pesos": amount_pesos,
                "client_name": client_name,
                "bank_name": self.bank
            })

        return r_entries


class DefaultBot:
    def __init__(self):
        pass

    @staticmethod
    def get_greeting(entry):
        hour = entry['system_current_datetime_object'].hour
        if 5 <= hour < 12:
            return "Buenos días"
        elif 12 <= hour <= 18:
            return "Buenas tardes"
        elif 18 < hour < 22:
            return "Good evening"
        else:
            return "Buenas noches"

    # gives the singular or plural of the days late
    @staticmethod
    def plural_singular_day(entry):
        if entry['days_late'] == 1:
            return "día"
        else:
            return "días"

    # gives the positive client identity confirmation
    @staticmethod
    def confirmation_with_gender(entry):
        if entry['client_name']['gender'] == "m":
            return ["si", "si, con el", "si el habla", "si con el"]
        elif entry['client_name']['gender'] == "f":
            return ["si", "si, con ella", "si ella habla", "si con ella"]

    # gives the negative client identity confirmation
    @staticmethod
    def negative_confirmation_gender(entry):
        if entry['client_name']['gender'] == "m":
            return ["no", "no señor", "no nabla con él"]
        elif entry['client_name']['gender'] == "f":
            return ["no", "no señor", "no habla con ella"]

    # returns the name of the line holder
    @staticmethod
    def line_holder_gender(entry):
        if entry['client_name']['gender'] == "m":
            return f"el titular de la línea es {entry['client_name']['name']}"
        elif entry['client_name']['gender'] == "f":
            return f"la titular de la línea es {entry['client_name']['name']}"

    # returns a negative reconfirmation about the identity of the line holder
    @staticmethod
    def negative_reconfirmation_gender(entry):
        if entry['client_name']['gender'] == "m":
            return ["no", "no conozco al señor que usted menciona", "no lo conozco",
                    f"no conozco al señor {entry['client_name']['name']}", "no se a quién busca",
                    "no se a quién se refiere"]
        elif entry['client_name']['gender'] == "f":
            return ["no", "no conozco a la señora que usted menciona", "no la conozco",
                    f"no conozco a la señora {entry['client_name']['name']}", "no se a quién busca",
                    "no se a quién se refiere"]


class AztecaGPTConversation:
    """
    Create random conversations using the Cobranzas Azteca bot flux and return the chat completion in OpenAI GPT format
    """

    def __init__(self, patterns=None, default_chat_flux=None, system_template=None):

        # search patterns inside assistant messages
        self.assistant_patterns = patterns or {
            "greeting": lambda content: "¿me comunico con" in content,
            "primary_info": lambda content: "saldo requerido para ponerla al día" in content,
            "amount_reconfirmation": lambda content: "Recuerda que el monto" in content,
            "final_reconfirmation": lambda content: "Genial, has confirmado tu pago." in content,
            "ask_for_tomorrow_pay": lambda content: "¿Podrías realizar el pago mañana" in content,
            "contact_you_later": lambda content: "Entendemos que no es el momento." in content,
            "first_attempt_agreement": lambda content: "Entendemos que puede ser complicado, pero si no" in content,
            "second_attempt_agreement": lambda content: "Si no pagas hoy podrías recibir molestias" in content,
            "ask_for_line_holder": lambda content: "¿Conoces al titular" in content,
            "ask_for_callback": lambda content: "¿Podrías pedirle que se comunique con nosotros" in content,
            "wrong_person": lambda content: "Gracias por tu tiempo y atención Lamento la confusión" in content
        }

        self.default_chat_flux = default_chat_flux or [
            [True, True, True, False, False, False, False, False, False, False, False],
            [True, True, False, False, False, False, False, False, False, False, False],
            [True, True, False, True, False, False, False, False, False, False, False],
            [True, False, False, False, True, True, False, False, False, False, False],
            [True, False, False, False, True, False, False, False, False, False, False],
            [True, False, False, False, True, False, True, False, False, False, False],
            [True, False, False, False, False, False, False, True, True, False, False],
            [True, False, False, False, False, False, False, True, False, False, False],
            [True, False, False, False, False, False, False, True, False, True, False],
            [True, False, False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False, False, True],
            [False, False, False, False, False, False, False, False, False, False, False]]

        self.system_template = system_template or (
            "Tu nombre es $agent y trabajas para Cumplir SAS. " 
            "Tu tarea es comunicarte con los clientes con alta empatía y comprensión. "
            "Nombre Banco: $bank_name. "
            "Nombre Cliente: $client_name. " 
            "Monto Adeudado: $amount pesos mexicanos. "
            "Fecha y hora de hoy: $today_date. "
            "Días de atraso en el pago: $days_late. " 
            "Fecha de pago máxima: $tomorrow_date."
        )

    # generates the conversation for the client negative identity
    @staticmethod
    def negative_identity_confirmation_block(entry, a):
        identity_confirmation = [{"role": "user", "content": random.choice(DefaultBot.negative_confirmation_gender(entry))},
                                 {"role": "assistant", "content": "¿Conoces al titular de la línea?"}]
        if a:
            identity_confirmation.append({"role": "user", "content": random.choice(
                ["si", "si le conozco", "si se a quién se refiere", f"{DefaultBot.line_holder_gender(entry)}"])})
            identity_confirmation.append({"role": "assistant",
                                          "content": "¿Podrías pedirle que se comunique con nosotros lo antes posible al 4775006675? Estamos disponibles de Lunes a domingo de 08:00am a 09:00pm. Agradecemos mucho tu ayuda. ¡Que tengas un buen día!"})
        else:
            identity_confirmation.append(
                {"role": "user", "content": random.choice(DefaultBot.negative_reconfirmation_gender(entry))})
            identity_confirmation.append({"role": "assistant",
                                          "content": "Gracias por tu tiempo y atención Lamento la confusión, parece que no eres la persona que estamos buscando. Agradezco tu ayuda ¡Que tengas un excelente día!"})

        return identity_confirmation

    @staticmethod
    def corrected_reconfirmation_block(entry, a, b):
        reconfirmation = [{"role": "assistant",
                           "content": f"Recuerda que el monto a pagar es {entry['amount_pesos']} pesos para el día {entry['current_date']}. ¿Confirmamos tu pago completo?"}]
        if a:
            reconfirmation.append({"role": "user", "content": random.choice(
                ["si, confirmo el pago", "confirmo el pago completo para el día de hoy", "confirmo el pago",
                 "confirmo el pago para esa fecha", "confirmo para hoy el pago", "si, para hoy"])})
            reconfirmation.append({"role": "assistant",
                                   "content": "Genial, has confirmado tu pago. Puedes hacerlo en la app, sucursal o aliados como Wallmart. ¡Gracias por tu compromiso!"})
        else:
            reconfirmation.append({"role": "user", "content": random.choice(
                ["no estoy seguro la verdad", "no se si pueda pagar hoy", "ahora que lo pensé la verdad no puedo",
                 "no creo poder hoy", "no puedo tener mas tiempo?", "me puedes dar plazo para completar?",
                 f"puedo pagar el {entry['ask_for_payment_day']}?"])})
            reconfirmation.append({"role": "assistant",
                                   "content": f"¿Podrías realizar el pago mañana {entry['tomorrow_date']} de tu monto requerido?"})
            if b:
                reconfirmation.append({"role": "user", "content": random.choice(
                    ["si", "si claro", f"el {entry['ask_for_payment_day']} si puedo",
                     "si claro no hay problema para mañana", "si, para mañana si puedo", "si puedo para mañana gracias",
                     "gracias por el plazo, para mañana si puedo pagar"])})
                reconfirmation.append({"role": "assistant",
                                       "content": "Genial, has confirmado tu pago. Puedes hacerlo en la app, sucursal o aliados como Wallmart. ¡Gracias por tu compromiso!"})
            else:
                reconfirmation.append({"role": "user", "content": random.choice(
                    ["no", "definitivamente no puedo tampoco para mañana",
                     "ni siquiera para mañana puedo necesito más plazo", "no, no puede darme mas plazo?",
                     "no puedo para mañana", "no podría ser para la siguiente semana?"])})
                reconfirmation.append({"role": "assistant",
                                       "content": "Entendemos que no es el momento. Nos comunicamos más tarde. ¡Feliz día!"})

        return reconfirmation

    # generates the system prompt for the fine tunning

    def system_block(self, entry):
        # system_str = (
        #     f"Tu nombre es {entry['name_of_the_agent']} y trabajas para Cumplir SAS. "
        #     f"Tu tarea es comunicarte con los clientes con alta empatía y comprensión. "
        #     f"Nombre Banco: {entry['bank_name']}. "
        #     f"Nombre Cliente: {entry['client_name']['name']}. "
        #     f"Monto Adeudado: {entry['amount_pesos']} pesos mexicanos. "
        #     f"Fecha y hora de hoy: {entry['system_current_date_time']}. "
        #     f"Días de atraso en el pago: {entry['days_late']}. "
        #     f"Fecha de pago máxima: {entry['system_tomorrow_date']}. "
        # )
        # return system_str

        sub = {
            'agent': entry['name_of_the_agent'],
            'days_late': entry['days_late'],
            'bank_name': entry['bank_name'],
            'client_name': entry['client_name']['name'],
            'amount': entry['amount_pesos'],
            'today_date': entry['system_current_date_time'],
            'tomorrow_date': entry['system_tomorrow_date'],
        }

        return Template(self.system_template).substitute(sub)

    # Generate a conversation based on the desired chatbot flux
    def generate_chat_completion(self, flux, entry):
        conversation = [
            {"role": "system", "content": self.system_block(entry)},
            {"role": "user", "content": random.choice(["hola", "aló?", "buenas", "si bueno", "bueno?"])},
            {"role": "assistant",
             "content": f"{DefaultBot.get_greeting(entry)}, ¿me comunico con {entry['client_name']['name']}?"}
        ]

        # identity confirmation
        if flux[0]:
            conversation.append({"role": "user", "content": random.choice(DefaultBot.confirmation_with_gender(entry))})
            conversation.append({"role": "assistant",
                                 "content": f"Perfecto, soy {entry['name_of_the_agent']} de {entry['bank_name']} y estamos contactándole para informarte que tienes {entry['days_late']} {DefaultBot.plural_singular_day(entry)} de atraso en tu cuenta y el saldo requerido para ponerla al día es de {entry['amount_pesos']} pesos. ¿Contamos con tu pago el día de hoy?"
                                 })

            # User response to payment request
            if flux[1]:
                # Client agrees to pay today
                conversation.append({"role": "user", "content": random.choice(
                    ["Si", "si, Voy a pagar hoy", "si, cuente con el pago el día de hoy"])})
                conversation.extend(self.corrected_reconfirmation_block(entry, flux[2], flux[3]))
            else:
                # Client cannot pay today, enter negotiation
                conversation.append({"role": "user", "content": random.choice(
                    ["No, no puedo para hoy", "no tengo tiempo para realizar el pago hoy"])})
                conversation.append({"role": "assistant",
                                     "content": f"Entendemos que puede ser complicado, pero si no regularizas tu pago hoy, podrías enfrentar mayores cargos o restricciones. ¿Podemos confirmar tu pago el día de hoy {entry['current_date']}?"})

                if flux[4]:
                    # Client agrees to pay
                    conversation.append(
                        {"role": "user", "content": random.choice(["ok, voy a pagar hoy", "esta bien, pago hoy"])})
                    conversation.extend(self.corrected_reconfirmation_block(entry, flux[5], flux[6]))
                else:
                    # Client still cannot pay
                    conversation.append({"role": "user", "content": random.choice(
                        ["definitivamente no puedo", "no tengo acceso a un punto de pago el día de hoy",
                         "puedo pagar otro día"])})
                    conversation.append({"role": "assistant",
                                         "content": "Si no pagas hoy podrías recibir molestias en tus teléfonos y cargos adicionales. Te invitamos a resolverlo ahora. ¿Confirmamos tu pago hoy?"})

                    if flux[7]:
                        conversation.append({"role": "user", "content": random.choice(
                            ["ok, voy a pagar hoy", "esta bien, ya voy a realizar el pago"])})
                        conversation.extend(self.corrected_reconfirmation_block(entry, flux[8], flux[9]))
                    else:
                        conversation.append({"role": "user", "content": random.choice(
                            ["no puedo", "definitivamente no puedo pagar hoy"])})
                        conversation.append({"role": "assistant",
                                             "content": "Entendemos que no es el momento. Nos comunicamos más tarde. ¡Feliz día!"})

        else:
            conversation.extend(self.negative_identity_confirmation_block(entry, flux[10]))

        conv_dict = {"messages": conversation}

        return conv_dict

class ConversationNoise:
    def __init__(self, conversation):
        self.conversation = conversation
    @staticmethod
    def add_noise():
        pass



class FineTuningDataset:

    def __init__(self):
        pass

    @staticmethod
    def add_weight_to_assistant_messages(conversation, default_weight=1):
        """
        Add a weight field to all "assistant" roles in the conversation.
        """
        for message in conversation["messages"]:
            if message["role"] == "assistant":
                message["weight"] = default_weight
        return conversation

    @staticmethod
    def create_dataset(entries, gpt_conv, chat_flux, generate_weights=False, filename="dataset3.jsonl"):
        """
        For each entry example generate a conversation with each chat flux and create a .jsonl file
        :param generate_weights: add weights to the assistant message
        :param entries: List of dictionaries with the conversation data
        :param chat_flux: List of lists with the chat flux config
        :param gpt_conv: GPTConversation instance
        :param filename: Name of the jsonl file

        """
        with open(filename, 'w', encoding='utf-8') as f:
            for entry in entries:
                for flux in chat_flux:
                    conversation = gpt_conv.generate_chat_completion(flux, entry)
                    if generate_weights:
                        FineTuningDataset.add_weight_to_assistant_messages(conversation)
                    f.write(json.dumps(conversation, ensure_ascii=False) + "\n")
        print(f"Dataset created successfully and saved in {filename}")

    @staticmethod
    def adjust_weights_by_line(input_file, output_file, thresholds, patterns):
        """
        Adjust weights in a .jsonl dataset based on line percentages.
        :param patterns:
        :param input_file: Path to the input .jsonl dataset.
        :param output_file: Path to save the modified .jsonl dataset.
        :param thresholds: Dictionary specifying the percentage threshold for each response type.
            Example: {"greeting": 0.5, "primary_info": 0.7}
        """
        # Read all lines to calculate total count
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        total_lines = len(lines)
        thresholds_count = {
            key: int(total_lines * threshold) for key, threshold in thresholds.items()
        }

        def assign_weight(line_number, part_name):
            """Determine if the weight for a specific part should be 1 or 0."""
            return 1 if line_number < thresholds_count[part_name] else 0

        # Write the modified dataset
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for line_number, line in enumerate(lines):
                conversation = json.loads(line)
                for message in conversation["messages"]:
                    if message["role"] == "assistant" and "weight" in message:
                        # Determine the type of response
                        for part_name, pattern in patterns.items():
                            if pattern(message["content"]):
                                # Adjust the weight based on thresholds
                                message["weight"] = assign_weight(line_number, part_name)
                                break

                # Write the modified conversation to the output file
                outfile.write(json.dumps(conversation, ensure_ascii=False) + "\n")

        print(f"Dataset processed successfully. Saved to {output_file}.")

    @staticmethod
    def analyze_dataset_patterns(dataset_file_path, conv_patterns):
        """
        Analyzes the dataset to count occurrences of specified patterns in assistant messages.

        :param dataset_file_path: path to jsonl dataset
        :param conv_patterns: Dictionary of patterns to identify in assistant messages.
                         Format: {"pattern_name": lambda content: boolean_expression}
        :return: Dictionary with counts for each pattern.
        """
        # Load the provided dataset for analysis
        with open(dataset_file_path, 'r', encoding='utf-8') as file:
            dataset = [json.loads(line) for line in file]

        pattern_counts = {pattern_name: 0 for pattern_name in conv_patterns}

        # Iterate over conversations and messages
        for conversation in dataset:
            for message in conversation.get("messages", []):
                if message.get("role") == "assistant":
                    content = message.get("content", "")
                    for pattern_name, pattern_func in conv_patterns.items():
                        if pattern_func(content):
                            pattern_counts[pattern_name] += 1

        return pattern_counts


if __name__ == "__main__":
    cli_manager = InputManager("client_names.jsonl")  # loads the entire dataset
    cli_data = cli_manager.get_sample(absolute=2)  # retrieve 20 random names

    # random dates start
    s_date = datetime(2024, 1, 1)

    # random dates end
    e_date = datetime.now()

    agent_names = ["Raúl"]
    bank_name = "Banco Azteca"

    dates_generator = DatesGenerator()

    test_entries_generator = RandomEntryGenerator(
        mode="current_day",
        ag_names=agent_names,
        cli_names=cli_data,
        bank=bank_name,
        date_config=dates_generator
    )

    test_entries = test_entries_generator.generate_random_entries(
        s_date,
        e_date
    )

    bot = AztecaGPTConversation()

    FineTuningDataset.create_dataset(
        entries=test_entries,
        gpt_conv=bot,
        chat_flux=bot.default_chat_flux,
        generate_weights=False,  # all the weights are set to 1 by default
        filename="test_azteca_v4.jsonl"
    )

    # count_train_patterns = FineTuningDataset.analyze_dataset_patterns(
    #     dataset_file_path="train_azteca_v1.jsonl",
    #     conv_patterns=bot.assistant_patterns
    # )
    #
    # print(count_train_patterns)
    #
    # assistant_thresholds_config = {
    #     "greeting": 0.5,  # After 50% of the dataset, set weight to 0. Set the 50% of the dataset with weight = 0
    #     "primary_info": 0.7,  # After 70% of the dataset, set weight to 0. Set the 30% of the dataset with weight = 0
    #     "amount_reconfirmation": 1,
    #     # After 100% of the dataset, set weight to 0. Set the 0% of the dataset with weight = 0
    #     "final_reconfirmation": 1,
    #     "ask_for_tomorrow_pay": 1,
    #     "contact_you_later": 1,
    #     "first_attempt_agreement": 1,
    #     "second_attempt_agreement": 1,
    #     "ask_for_line_holder": 1,
    #     "ask_for_callback": 1,
    #     "wrong_person": 1
    # }
    #
    # FineTuningDataset.adjust_weights_by_line(
    #     input_file="train_azteca_v1.jsonl",
    #     output_file="train_azteca_weights_v1.jsonl",
    #     thresholds=assistant_thresholds_config,
    #     patterns=bot.assistant_patterns
    # )
