# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Testings for the clone functionality of agents."""

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent


def test_llm_agent_clone():
  """Test cloning an LLM agent."""
  # Create an LLM agent
  original = LlmAgent(
      name="llm_agent",
      description="An LLM agent",
      instruction="You are a helpful assistant.",
  )

  # Clone it
  cloned = original.clone("cloned_llm_agent")

  # Verify the clone
  assert cloned.name == "cloned_llm_agent"
  assert cloned.description == "An LLM agent"
  assert cloned.instruction == "You are a helpful assistant."
  assert cloned.parent_agent is None
  assert len(cloned.sub_agents) == 0
  assert isinstance(cloned, LlmAgent)

  # Verify the original is unchanged
  assert original.name == "llm_agent"
  assert original.instruction == "You are a helpful assistant."


def test_agent_with_sub_agents():
  """Test cloning an agent that has sub-agents."""
  # Create sub-agents
  sub_agent1 = LlmAgent(name="sub_agent1", description="First sub-agent")

  sub_agent2 = LlmAgent(name="sub_agent2", description="Second sub-agent")

  # Create a parent agent with sub-agents
  original = SequentialAgent(
      name="parent_agent",
      description="Parent agent with sub-agents",
      sub_agents=[sub_agent1, sub_agent2],
  )

  # Clone it
  cloned = original.clone("cloned_parent")

  # Verify the clone has no sub-agents
  assert cloned.name == "cloned_parent"
  assert cloned.description == "Parent agent with sub-agents"
  assert cloned.parent_agent is None
  assert len(cloned.sub_agents) == 2
  assert cloned.sub_agents[0].name == "sub_agent1_clone"
  assert cloned.sub_agents[1].name == "sub_agent2_clone"

  # Verify the original still has sub-agents
  assert original.name == "parent_agent"
  assert len(original.sub_agents) == 2
  assert original.sub_agents[0].name == "sub_agent1"
  assert original.sub_agents[1].name == "sub_agent2"


def test_multiple_clones():
  """Test creating multiple clones with automatic naming."""
  # Create multiple agents and clone each one
  original = LlmAgent(
      name="original_agent", description="Agent for multiple cloning"
  )

  # Test multiple clones from the same original
  clone1 = original.clone()
  clone2 = original.clone()

  assert clone1.name == "original_agent_clone"
  assert clone2.name == "original_agent_clone"
  assert clone1 is not clone2


def test_clone_with_complex_configuration():
  """Test cloning an agent with complex configuration."""
  # Create an LLM agent with various configurations
  original = LlmAgent(
      name="complex_agent",
      description="A complex agent with many settings",
      instruction="You are a specialized assistant.",
      global_instruction="Always be helpful and accurate.",
      disallow_transfer_to_parent=True,
      disallow_transfer_to_peers=True,
      include_contents="none",
  )

  # Clone it
  cloned = original.clone("complex_clone")

  # Verify all configurations are preserved
  assert cloned.name == "complex_clone"
  assert cloned.description == "A complex agent with many settings"
  assert cloned.instruction == "You are a specialized assistant."
  assert cloned.global_instruction == "Always be helpful and accurate."
  assert cloned.disallow_transfer_to_parent is True
  assert cloned.disallow_transfer_to_peers is True
  assert cloned.include_contents == "none"

  # Verify parent and sub-agents are set
  assert cloned.parent_agent is None
  assert len(cloned.sub_agents) == 0


def test_clone_without_name():
  """Test cloning without providing a name (should use default naming)."""
  original = LlmAgent(name="test_agent", description="Test agent")

  cloned = original.clone()

  assert cloned.name == "test_agent_clone"
  assert cloned.description == "Test agent"


def test_clone_preserves_agent_type():
  """Test that cloning preserves the specific agent type."""
  # Test LlmAgent
  llm_original = LlmAgent(name="llm_test")
  llm_cloned = llm_original.clone()
  assert isinstance(llm_cloned, LlmAgent)

  # Test SequentialAgent
  seq_original = SequentialAgent(name="seq_test")
  seq_cloned = seq_original.clone()
  assert isinstance(seq_cloned, SequentialAgent)

  # Test ParallelAgent
  par_original = ParallelAgent(name="par_test")
  par_cloned = par_original.clone()
  assert isinstance(par_cloned, ParallelAgent)

  # Test LoopAgent
  loop_original = LoopAgent(name="loop_test")
  loop_cloned = loop_original.clone()
  assert isinstance(loop_cloned, LoopAgent)
