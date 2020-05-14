import smtplib

# TODO do not lehagish

def send_email(username, password, to_addresses, body, subject='Your code just finished executing'):
    """
    Sends an email from a gmail account to any other email account, saying the code has finished executing.
    Note: in order for this to work you have to go to your gmail settings and turn on "less secure apps" setting.
    :param username: Your gmail login username
    :param password: Your gmail password
    :param to_addresses: List of strings with recipients addresses
    :param body: Body of the message. It is recommended to include start and end times of the code execution.
    :param subject: Subject of the message. Defaults to 'Your code just finished executing'.
    """
    msg = """From: Python gods\nTo: %s\nSubject: %s\n\n%s""" % (", ".join(to_addresses), subject, body)
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.ehlo()
    server.login(username, password)
    server.sendmail(username, to_addresses, msg)
    server.close()


def notify_email(start_time, preprocess_time, optimization_time, prediction_time):
    # TODO put this to use
    message_body = 'Start: ' + start_time + '\nPreprocess end: ' + preprocess_time + '\nOptimization end: ' + \
                   optimization_time + '\nPrediction end: ' + prediction_time
    send_email('eldar.abraham@gmail.com', '<>', ['eldar.a@campus.technion.ac.il'], message_body)