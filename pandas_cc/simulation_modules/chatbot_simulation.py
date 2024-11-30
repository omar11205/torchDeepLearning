import random
import json
import locale
from datetime import datetime, timedelta

# reproducibility
random.seed(42)

# dates in spanish
locale.setlocale(locale.LC_ALL, "es_ES.UTF-8")


# load client names
def load_client_names(filename="/client_names.jsonl"):
    """
    Loads client names from a jsonl file in the format {client_name: client_name, gender: gender}
    :param filename:
    :return: list of client names dictionaries
    """
    with open(filename, "r", encoding="utf-8") as file:
        client_names = [json.loads(line) for line in file]
    return client_names


# Aux. Function to find duplicated names
def find_duplicates(client_list):
    seen_names = set()
    duplicate_names = set()

    for client in client_list:
        name = client["name"]
        if name in seen_names:
            duplicate_names.add(name)
        else:
            seen_names.add(name)

    return duplicate_names


# print duplicated names
def show_duplicates(cli_names):
    duplicates = find_duplicates(cli_names)
    if duplicates:
        print("Total duplicated names:", len(duplicates))
        print("Duplicate names found:", duplicates)
    else:
        print("No duplicate names found.")


# Function to verify business dates
def is_business_day(date, holidays=None):
    holidays = holidays or []
    return date.weekday() < 5 and date not in holidays


# Function to generate random business dates
def generate_random_business_dates(str_date, e_date, n_dates, holidays=None):
    holidays = holidays or []
    business_days = []

    # Create a list of all dates between start_date and end_date
    current_date = str_date
    while current_date <= e_date:
        if is_business_day(current_date, holidays):
            business_days.append(current_date)
        current_date += timedelta(days=1)

    # Select n_dates random dates
    if n_dates > len(business_days):
        raise ValueError("No hay suficientes días hábiles en el rango especificado.")

    return random.sample(business_days, n_dates)


def generate_random_business_datetimes(s_date, e_date, n_dates, holidays=None):
    holidays = holidays or []
    business_days = []

    # Create a list of all business dates between start_date and end_date
    current_date = s_date
    while current_date <= e_date:
        if is_business_day(current_date, holidays):
            business_days.append(current_date)
        current_date += timedelta(days=1)

    # Select n_dates random business dates
    if n_dates > len(business_days):
        raise ValueError("No hay suficientes días hábiles en el rango especificado.")

    random_dates = random.sample(business_days, n_dates)

    # Generate random times for each selected date within the specified range
    business_datetimes = []
    for date in random_dates:
        random_hour = random.randint(7, 18)  # 7 AM to 6 PM (inclusive)
        random_minute = random.randint(0, 59)
        random_second = random.randint(0, 59)
        business_datetime = datetime.combine(date, datetime.min.time()) + timedelta(
            hours=random_hour, minutes=random_minute, seconds=random_second
        )
        business_datetimes.append(business_datetime)

    return business_datetimes


# Generate random entries for the chatbot simulation
def generate_random_entries(ag_names, cli_names, str_date, e_date, holidays, bank_name, n_dates=1):
    random.shuffle(cli_names)
    entries = []

    for name in cli_names:
        # Randomly choose an agent and client name
        name_of_the_agent = random.choice(ag_names)
        client_name = name

        # Generate a random number of days late and corresponding amount in pesos
        days_late = random.randint(1, 60)  # e.g., 1 to 60 days late
        amount_pesos = random.randint(1000, 50000)  # e.g., 1000 to 50000 pesos

        # Generate current date and tomorrow's date
        current_date = datetime.now().strftime("%d de %B de %Y")  # E.g., "14 de Noviembre de 2024"
        tomorrow_date = (datetime.now() + timedelta(days=1)).strftime("%d de %B de %Y")

        # Generate system current and tomorrow's date
        random_non_holiday = generate_random_business_datetimes(str_date, e_date, n_dates, holidays)[0]

        # Manually construct the AM/PM part
        am_pm = "AM" if random_non_holiday.hour < 12 else "PM"

        system_current_date_time = random_non_holiday.strftime(f"%A %Y-%m-%d %I:%M {am_pm}")  # E.g: "viernes 2024-11-16"
        system_tomorrow_date = (random_non_holiday + timedelta(days=1)).strftime("%A %Y-%m-%d") #E.g "sábado 2024-11-17"
        ask_for_payment_day = (random_non_holiday + timedelta(days=1)).strftime("%A")

        # Append each generated entry as a dictionary
        entries.append({
            "ask_for_payment_day": ask_for_payment_day,
            "system_current_date_time": system_current_date_time,
            "system_current_datetime_object": random_non_holiday,
            "system_tomorrow_date": system_tomorrow_date,
            "current_date": current_date,
            "tomorrow_date": tomorrow_date,
            "name_of_the_agent": name_of_the_agent,
            "days_late": days_late,
            "amount_pesos": amount_pesos,
            "client_name": client_name,
            "bank_name": bank_name
        })

    return entries


# Define subroutine for reconfirmation block
def corrected_reconfirmation_block(entry, a, b):
      reconfirmation = []
      reconfirmation.append({"role": "assistant", "content": f"Recuerda que el monto a pagar es {entry['amount_pesos']} pesos para el día {entry['current_date']}. ¿Confirmamos tu pago completo?"})
      if a:
          reconfirmation.append({"role": "user", "content": random.choice(["si, confirmo el pago", "confirmo el pago completo para el día de hoy", "confirmo el pago", "confirmo el pago para esa fecha", "confirmo para hoy el pago", "si, para hoy"])})
          reconfirmation.append({"role": "assistant", "content": "Genial, has confirmado tu pago. Puedes hacerlo en la app, sucursal o aliados como Wallmart. ¡Gracias por tu compromiso!"})
      else:
          reconfirmation.append({"role": "user", "content": random.choice(["no estoy seguro la verdad", "no se si pueda pagar hoy", "ahora que lo pensé la verdad no puedo", "no creo poder hoy", "no puedo tener mas tiempo?", "me puedes dar plazo para completar?", f"puedo pagar el {entry['ask_for_payment_day']}?"])})
          reconfirmation.append({"role": "assistant", "content": f"¿Podrías realizar el pago mañana {entry['tomorrow_date']} de tu monto requerido?"})
          if b:
              reconfirmation.append({"role": "user", "content": random.choice(["si", "si claro", f"el {entry['ask_for_payment_day']} si puedo", "si claro no hay problema para mañana", "si, para mañana si puedo", "si puedo para mañana gracias", "gracias por el plazo, para mañana si puedo pagar"])})
              reconfirmation.append({"role": "assistant", "content": "Genial, has confirmado tu pago. Puedes hacerlo en la app, sucursal o aliados como Wallmart. ¡Gracias por tu compromiso!"})
          else:
              reconfirmation.append({"role": "user", "content": random.choice(["no", "definitivamente no puedo tampoco para mañana", "ni siquiera para mañana puedo necesito más plazo", "no, no puede darme mas plazo?", "no puedo para mañana", "no podría ser para la siguiente semana?"])})
              reconfirmation.append({"role": "assistant", "content": "Entendemos que no es el momento. Nos comunicamos más tarde. ¡Feliz día!"})

      return reconfirmation


# return the daytime greeting
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
def plural_singular_day(entry):
    if entry['days_late'] == 1:
        return "día"
    else:
        return "días"


# gives the positive client identity confirmation
def confirmation_with_gender(entry):
    if entry['client_name']['gender'] == "m":
        return ["si", "si, con el", "si el habla", "si con el"]
    elif entry['client_name']['gender'] == "f":
        return ["si", "si, con ella", "si ella habla", "si con ella"]


# gives the negative client identity confirmation
def negative_confirmation_gender(entry):
    if entry['client_name']['gender'] == "m":
        return ["no", "no señor", "no nabla con él"]
    elif entry['client_name']['gender'] == "f":
        return ["no", "no señor", "no habla con ella"]


# returns the name of the line holder
def line_holder_gender(entry):
    if entry['client_name']['gender'] == "m":
        return f"el titular de la línea es {entry['client_name']['name']}"
    elif entry['client_name']['gender'] == "f":
        return f"la titular de la línea es {entry['client_name']['name']}"


# returns a negative reconfirmation about the identity of the line holder
def negative_reconfirmation_gender(entry):
    if entry['client_name']['gender'] == "m":
        return ["no", "no conozco al señor que usted menciona", "no lo conozco", f"no conozco al señor {entry['client_name']['name']}", "no se a quién busca", "no se a quién se refiere"]
    elif entry['client_name']['gender'] == "f":
        return ["no", "no conozco a la señora que usted menciona", "no la conozco", f"no conozco a la señora {entry['client_name']['name']}", "no se a quién busca", "no se a quién se refiere"]


# generates the conversation for the client negative identity
def negative_identity_confirmation_block(entry, a):
    identity_confirmation = []
    identity_confirmation.append({"role": "user", "content": random.choice(negative_confirmation_gender(entry))})
    identity_confirmation.append({"role": "assistant", "content": "¿Conoces al titular de la línea?"})
    if a:
        identity_confirmation.append({"role": "user", "content": random.choice(["si", "si le conozco", "si se a quién se refiere", f"{line_holder_gender(entry)}"])})
        identity_confirmation.append({"role": "assistant", "content": "¿Podrías pedirle que se comunique con nosotros lo antes posible al 4775006675? Estamos disponibles de Lunes a domingo de 08:00am a 09:00pm. Agradecemos mucho tu ayuda. ¡Que tengas un buen día!"})
    else:
        identity_confirmation.append({"role": "user", "content": random.choice(negative_reconfirmation_gender(entry))})
        identity_confirmation.append({"role": "assistant", "content": "Gracias por tu tiempo y atención Lamento la confusión, parece que no eres la persona que estamos buscando. Agradezco tu ayuda ¡Que tengas un excelente día!"})

    return identity_confirmation


# generates the system prompt for the fine tunning
def system_block(entry):
    system_str = (
      f"Tu nombre es {entry['name_of_the_agent']} y trabajas para Cumplir SAS. "
      f"Tu tarea es comunicarte con los clientes con alta empatía y comprensión. "
      f"Nombre Banco: {entry['bank_name']}. "
      f"Nombre Cliente: {entry['client_name']['name']}. "
      f"Monto Adeudado: {entry['amount_pesos']} pesos mexicanos. "
      f"Fecha y hora de hoy: {entry['system_current_date_time']}. "
      f"Días de atraso en el pago: {entry['days_late']}. "
      f"Fecha de pago máxima: {entry['system_tomorrow_date']}. "
    )
    return system_str


default_flux = [True for _ in range(11)]
# voice-bot simulation

# Generate a conversation based on the desired chatbot flux
def generate_conversation(entry, flux=default_flux, assistant_weight=1):
    conversation = [
        {"role": "system", "content": system_block(entry)},
        {"role": "user", "content": random.choice(["hola", "aló", "buenas", "bueno"])},
        {"role": "assistant", "content": f"{get_greeting(entry)}, ¿me comunico con {entry['client_name']['name']}?", "weight": assistant_weight}
    ]

    # identity confirmation
    if flux[0]:
        conversation.append({"role": "user", "content": random.choice(confirmation_with_gender(entry))})
        conversation.append({"role": "assistant", "content": f"Perfecto, soy {entry['name_of_the_agent']} de {entry['bank_name']} y estamos contactándole para informarte que tienes {entry['days_late']} {plural_singular_day(entry)} de atraso en tu cuenta y el saldo requerido para ponerla al día es de {entry['amount_pesos']} pesos. ¿Contamos con tu pago el día de hoy?", "weight": assistant_weight})

        # User response to payment request
        if flux[1]:
            # Client agrees to pay today
            conversation.append({"role": "user", "content": random.choice(["Si", "si, Voy a pagar hoy", "si, cuente con el pago el día de hoy"])})
            conversation.extend(corrected_reconfirmation_block(entry, flux[2], flux[3]))
        else:
            # Client cannot pay today, enter negotiation
            conversation.append({"role": "user", "content": random.choice(["No, no puedo para hoy", "no tengo tiempo para realizar el pago hoy"])})
            conversation.append({"role": "assistant", "content": f"Entendemos que puede ser complicado, pero si no regularizas tu pago hoy, podrías enfrentar mayores cargos o restricciones. ¿Podemos confirmar tu pago el día de hoy {entry['current_date']}?"})

            if flux[4]:
                # Client agrees to pay
                conversation.append({"role": "user", "content": random.choice(["ok, voy a pagar hoy", "esta bien, pago hoy"])})
                conversation.extend(corrected_reconfirmation_block(entry, flux[5], flux[6]))
            else:
                # Client still cannot pay
                conversation.append({"role": "user", "content": random.choice(["definitivamente no puedo", "no tengo acceso a un punto de pago el día de hoy", "puedo pagar otro día"])})
                conversation.append({"role": "assistant", "content": "Si no pagas hoy podrías recibir molestias en tus teléfonos y cargos adicionales. Te invitamos a resolverlo ahora. ¿Confirmamos tu pago hoy?"})

                if flux[7]:
                    conversation.append({"role": "user", "content": random.choice(["ok, voy a pagar hoy", "esta bien, ya voy a realizar el pago"])})
                    conversation.extend(corrected_reconfirmation_block(entry, flux[8], flux[9]))
                else:
                    conversation.append({"role": "user", "content": random.choice(["no puedo", "definitivamente no puedo pagar hoy"])})
                    conversation.append({"role": "assistant", "content": "Entendemos que no es el momento. Nos comunicamos más tarde. ¡Feliz día!"})

    else:
        conversation.extend(negative_identity_confirmation_block(entry, flux[10]))

    return {"messages": conversation}


def create_dataset_v1(sett, chat_flux, filename="dataset3.jsonl", shutdown_assistant=50):
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in sett:
            for flux in chat_flux:
                conversation = generate_conversation(entry, flux)
                f.write(json.dumps(conversation, ensure_ascii=False) + "\n")
    print("Dataset created successfully.")


def create_dataset_v2(sett, chat_flux, filename="dataset3.jsonl", percentage_threshold=50):
    """
    Create a fine-tuning dataset with dynamic assistant_weight adjustment.

    Parameters:
        sett: List of entries for conversations.
        chat_flux: List of flux configurations for conversations.
        filename: Name of the output file.
        percentage_threshold: Percentage of the dataset after which `assistant_weight` is set to 0.
    """
    total_entries = len(sett) * len(chat_flux)
    threshold_count = int(total_entries * (percentage_threshold / 100))  # Calculate the threshold index

    current_count = 0  # Track the number of entries processed

    with open(filename, 'w', encoding='utf-8') as f:
        for entry in sett:
            for flux in chat_flux:
                # Determine weight dynamically
                assistant_weight = 0 if current_count >= threshold_count else 1

                # Generate conversation
                conversation = generate_conversation(entry, flux, assistant_weight)

                # Write conversation to file
                f.write(json.dumps(conversation, ensure_ascii=False) + "\n")

                # Update count
                current_count += 1

    print("Dataset created successfully.")


def create_dataset_v3(sett, chat_flux, filename="dataset3.jsonl", weight_control=None):
    """
    Create a fine-tuning dataset with dynamic assistant_weight adjustment for specific parts of the conversation.

    Parameters:
        sett: List of entries for conversations.
        chat_flux: List of flux configurations for conversations.
        filename: Name of the output file.
        weight_control: Dictionary with thresholds for specific conversation parts.
                        Example: {"greeting": 0.5, "primary_info": 0.7}
    """
    if weight_control is None:
        weight_control = {"greeting": 0.5, "primary_info": 0.7}

    # Counters to track the weights applied
    counters = {key: 0 for key in weight_control.keys()}
    total_counts = {key: 0 for key in weight_control.keys()}

    def should_assign_weight(part_name):
        """Determine if the weight for a specific part should be 1."""
        nonlocal counters, total_counts
        total_counts[part_name] += 1
        if counters[part_name] / total_counts[part_name] < weight_control[part_name]:
            counters[part_name] += 1
            return 1
        return 0

    with open(filename, 'w', encoding='utf-8') as f:
        for entry in sett:
            for flux in chat_flux:
                # Assign weights dynamically to specific parts of the conversation
                greeting_weight = should_assign_weight("greeting")
                primary_info_weight = should_assign_weight("primary_info")

                # Generate conversation
                conversation = generate_conversation(
                    entry,
                    flux,
                    assistant_weight=greeting_weight,  # Use dynamic weight for "greeting"
                )

                # Manually adjust weights for specific parts
                for message in conversation["messages"]:
                    if "¿me comunico con" in message["content"]:
                        message["weight"] = greeting_weight
                    elif "Perfecto, soy" in message["content"]:
                        message["weight"] = primary_info_weight

                # Write conversation to file
                f.write(json.dumps(conversation, ensure_ascii=False) + "\n")

    print("Dataset created successfully with dynamic weights.")


def adjust_weights(input_file, output_file, weight_control):
    """
    Adjust weights in a .jsonl dataset for specific assistant responses.

    Parameters:
        input_file: Path to the input .jsonl dataset.
        output_file: Path to save the modified .jsonl dataset.
        weight_control: Dictionary with thresholds for specific response types.
                        Example: {"greeting": 0.5, "primary_info": 0.7}
    """
    # Counters to track the number of responses processed
    counters = {key: 0 for key in weight_control.keys()}
    total_counts = {key: 0 for key in weight_control.keys()}

    def should_assign_weight(part_name):
        """Determine if the weight for a specific part should be 1."""
        nonlocal counters, total_counts
        total_counts[part_name] += 1
        if counters[part_name] / total_counts[part_name] < weight_control[part_name]:
            counters[part_name] += 1
            return 1
        return 0

    # Patterns to classify responses
    patterns = {
        "greeting": lambda content: "¿me comunico con" in content,
        "primary_info": lambda content: "saldo requerido para ponerla al día" in content
    }

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            conversation = json.loads(line)
            for message in conversation["messages"]:
                if message["role"] == "assistant" and "weight" in message:
                    # Determine the type of response
                    for part_name, pattern in patterns.items():
                        if pattern(message["content"]):
                            # Adjust the weight based on thresholds
                            message["weight"] = should_assign_weight(part_name)
                            break

            # Write the modified conversation to the output file
            outfile.write(json.dumps(conversation, ensure_ascii=False) + "\n")

    print(f"Dataset processed successfully. Saved to {output_file}.")


def adjust_weights_by_line(input_file, output_file, thresholds):
    """
    Adjust weights in a .jsonl dataset based on line percentages.

    Parameters:
        input_file: Path to the input .jsonl dataset.
        output_file: Path to save the modified .jsonl dataset.
        thresholds: Dictionary specifying the percentage threshold for each response type.
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

    # Patterns to classify responses
    patterns = {
        "greeting": lambda content: "¿me comunico con" in content,
        "primary_info": lambda content: "saldo requerido para ponerla al día" in content
    }

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





if __name__ == "__main__":

    # random dates start
    start_date = datetime(2024, 1, 1)

    # random dates end
    end_date = datetime(2024, 12, 31)

    # define the agent name
    agent_name = ["Raúl"]

    # name of the bank
    b_name = "Banco Azteca"

    # Mexican holidays
    mexico_holidays = [
        datetime(2024, 1, 1),  # Año Nuevo
        datetime(2024, 2, 5),  # Día de la Constitución (primer lunes de febrero)
        datetime(2024, 3, 18),  # Natalicio de Benito Juárez (tercer lunes de marzo)
        datetime(2024, 5, 1),  # Día del Trabajo
        datetime(2024, 9, 16),  # Día de la Independencia
        datetime(2024, 11, 18),  # Revolución Mexicana (tercer lunes de noviembre)
        datetime(2024, 12, 25)  # Navidad
    ]

    # load 400 client names
    client_names = load_client_names("client_names.jsonl")

    # select a sample of client names
    clients_subset = random.sample(client_names, 6)

    # generate the entries for each client conversation
    entries = generate_random_entries(agent_name, clients_subset, start_date, end_date, mexico_holidays, b_name)

    # 12 possible conversation fluxes
    chatbot_flux_config = [[True, True, True, False, False, False, False, False, False, False, False],
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
                           [False, False, False, False, False, False, False, False, False, False, False]
                           ]

    chatbot_flux_config_2 = [[True, True, True, False, False, False, False, False, False, False, False]]

    # create an ordered dataset for the fine tunning
    create_dataset_v1(entries, chatbot_flux_config, filename="azteca_v3.jsonl")

    thresholds_config = {
        "greeting": 0.5,  # After 50% of the dataset, set weight to 0
        "primary_info": 0.7  # After 70% of the dataset, set weight to 0
    }

    adjust_weights_by_line(
        input_file="azteca_v3.jsonl",
        output_file="azteca_train_v3.jsonl",
        thresholds=thresholds_config
    )

    # creating test dataset
    test_clients_subset = random.sample(client_names, 2)
    test_entries = generate_random_entries(agent_name, test_clients_subset, start_date, end_date, mexico_holidays, b_name)
    create_dataset_v1(test_entries, chatbot_flux_config, filename="azteca_t_v3.jsonl")
    adjust_weights_by_line(
        input_file="azteca_t_v3.jsonl",
        output_file="azteca_test_v3.jsonl",
        thresholds=thresholds_config
    )

    # creating open_ai evaluation dataset
    test_clients_subset = random.sample(client_names, 50)
    test_entries = generate_random_entries(agent_name, test_clients_subset, start_date, end_date, mexico_holidays, b_name)
    create_dataset_v1(test_entries, chatbot_flux_config, filename="azteca_e_v3.jsonl")
    adjust_weights_by_line(
        input_file="azteca_e_v3.jsonl",
        output_file="azteca_evaluation_v3.jsonl",
        thresholds=thresholds_config
    )




