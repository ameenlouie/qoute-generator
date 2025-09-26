# quote_generator.py

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_quote_chain():
    """
    Creates and returns a LangChain chain for generating motivational quotes.
    """
    # 1. Connect to the local Ollama model
    # Make sure you have run 'ollama pull phi3' in your terminal
    llm = Ollama(model="phi3")

    # 2. Create a prompt template
    # This gives the AI its persona and instructions
    prompt_template = ChatPromptTemplate.from_template(
        "You are a world-renowned motivational speaker. "
        "Generate a single, short, and powerful motivational quote about the topic of '{topic}'. "
        "Do not provide any explanation or attribution. Just the quote."
    )

    # 3. Create a simple output parser to get a string output
    output_parser = StrOutputParser()

    # 4. Create the chain by piping the components together
    chain = prompt_template | llm | output_parser
    
    return chain

# --- Main execution loop ---
if __name__ == "__main__":
    print("✨ Motivational Quote Generator ✨")
    print("---------------------------------")
    
    # Create the generator chain
    quote_chain = create_quote_chain()
    
    while True:
        # Get topic from the user
        user_topic = input("Enter a topic (or type 'exit' to quit): ")
        
        if user_topic.lower() == 'exit':
            print("Keep striving! Goodbye.")
            break
            
        if not user_topic.strip():
            print("Please enter a valid topic.")
            continue
            
        print("\nGenerating your quote...\n")
        
        try:
            # Invoke the chain with the user's topic
            # The invoke method runs the chain and returns the final output
            quote = quote_chain.invoke({"topic": user_topic})
            
            # Print the generated quote
            print(f"Here is a quote about '{user_topic.capitalize()}':")
            print(f'"{quote.strip()}"\n')
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please ensure the Ollama application is running and the 'phi3' model is downloaded.")
            break
