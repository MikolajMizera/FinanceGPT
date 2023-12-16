from financegpt.llm.chain import LLMChainInterfaceFactory


def test_create_llm_chain():
    llm_chain = LLMChainInterfaceFactory.create_llm_chain("gpt-3.5-turbo-instruct")
    assert hasattr(llm_chain._chain.first, "model_name")
    assert llm_chain._chain.first.model_name == "gpt-3.5-turbo-instruct"  # type: ignore
