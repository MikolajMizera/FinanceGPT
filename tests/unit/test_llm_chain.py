from financegpt.llm.chain import LLMChainInterfaceFactory


def test_create_llm_chain():
    llm_chain = LLMChainInterfaceFactory.create_llm_chain("gpt-3.5-turbo")
    assert hasattr(llm_chain._chain.first, "model_name")
    assert llm_chain._chain.first.model_name == "gpt-3.5-turbo"  # type: ignore
