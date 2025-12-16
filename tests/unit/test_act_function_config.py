"""
Unit tests for ActFunctionConfig integration.

Tests that the configuration properly flows from BotConfig -> BrowserVisionBot -> AgentController
and that the agent controller correctly uses these settings when building act parameters.
"""

import pytest
from bot_config import BotConfig, ActFunctionConfig
from vision_bot import BrowserVisionBot
from agent.agent_controller import AgentController
from unittest.mock import MagicMock


class TestActFunctionConfig:
    """Test suite for ActFunctionConfig functionality"""
    
    def test_default_config_all_enabled(self):
        """Test that default config has all parameters enabled"""
        config = ActFunctionConfig()
        
        assert config.enable_target_context_guard is True
        assert config.enable_modifier is True
        assert config.enable_additional_context is True
    
    def test_config_with_target_context_guard_disabled(self):
        """Test config with target_context_guard disabled"""
        config = ActFunctionConfig(
            enable_target_context_guard=False,
            enable_modifier=True,
            enable_additional_context=True
        )
        
        assert config.enable_target_context_guard is False
        assert config.enable_modifier is True
        assert config.enable_additional_context is True
    
    def test_config_with_modifier_disabled(self):
        """Test config with modifier disabled"""
        config = ActFunctionConfig(
            enable_target_context_guard=True,
            enable_modifier=False,
            enable_additional_context=True
        )
        
        assert config.enable_target_context_guard is True
        assert config.enable_modifier is False
        assert config.enable_additional_context is True
    
    def test_config_with_additional_context_disabled(self):
        """Test config with additional_context disabled"""
        config = ActFunctionConfig(
            enable_target_context_guard=True,
            enable_modifier=True,
            enable_additional_context=False
        )
        
        assert config.enable_target_context_guard is True
        assert config.enable_modifier is True
        assert config.enable_additional_context is False
    
    def test_config_all_disabled(self):
        """Test config with all parameters disabled"""
        config = ActFunctionConfig(
            enable_target_context_guard=False,
            enable_modifier=False,
            enable_additional_context=False
        )
        
        assert config.enable_target_context_guard is False
        assert config.enable_modifier is False
        assert config.enable_additional_context is False
    
    def test_bot_config_includes_act_function_config(self):
        """Test that BotConfig properly includes ActFunctionConfig"""
        bot_config = BotConfig(
            act_function=ActFunctionConfig(
                enable_target_context_guard=False,
                enable_modifier=True,
                enable_additional_context=True
            )
        )
        
        assert bot_config.act_function.enable_target_context_guard is False
        assert bot_config.act_function.enable_modifier is True
        assert bot_config.act_function.enable_additional_context is True
    
    def test_vision_bot_extracts_act_function_config(self):
        """Test that BrowserVisionBot extracts act function config settings"""
        config = BotConfig(
            act_function=ActFunctionConfig(
                enable_target_context_guard=False,
                enable_modifier=True,
                enable_additional_context=False
            )
        )
        
        # Create a mock page to avoid browser initialization
        mock_page = MagicMock()
        bot = BrowserVisionBot(config=config, page=mock_page)
        
        # Verify the settings were extracted
        assert hasattr(bot, 'act_enable_target_context_guard')
        assert hasattr(bot, 'act_enable_modifier')
        assert hasattr(bot, 'act_enable_additional_context')
        
        assert bot.act_enable_target_context_guard is False
        assert bot.act_enable_modifier is True
        assert bot.act_enable_additional_context is False
    
    def test_agent_controller_receives_config(self):
        """Test that AgentController receives and stores act function config"""
        from ai_utils import ReasoningLevel
        
        # Create a mock bot with act function config
        mock_bot = MagicMock()
        mock_bot.act_enable_target_context_guard = False
        mock_bot.act_enable_modifier = True
        mock_bot.act_enable_additional_context = False
        mock_bot.agent_model_name = "gpt-5-mini"
        mock_bot.agent_reasoning_level = ReasoningLevel.NONE
        mock_bot.event_logger = MagicMock()
        
        # Create agent controller
        controller = AgentController(
            mock_bot,
            act_enable_target_context_guard=False,
            act_enable_modifier=True,
            act_enable_additional_context=False
        )
        
        # Verify the settings were stored
        assert controller.act_enable_target_context_guard is False
        assert controller.act_enable_modifier is True
        assert controller.act_enable_additional_context is False
    
    def test_extract_act_params_respects_target_context_guard_disabled(self):
        """Test that _parse_action_for_act_params respects disabled target_context_guard"""
        from ai_utils import ReasoningLevel
        
        mock_bot = MagicMock()
        mock_bot.agent_model_name = "gpt-5-mini"
        mock_bot.agent_reasoning_level = ReasoningLevel.NONE
        mock_bot.event_logger = MagicMock()
        
        # Create controller with target_context_guard disabled
        controller = AgentController(
            mock_bot,
            act_enable_target_context_guard=False,
            act_enable_modifier=True,
            act_enable_additional_context=True
        )
        
        # Test with an action that would normally create a target_context_guard
        params = controller._parse_action_for_act_params("click: first button", "test prompt")
        
        # target_context_guard should be None even though "first" was specified
        assert params["target_context_guard"] is None
    
    def test_extract_act_params_respects_modifier_disabled(self):
        """Test that _parse_action_for_act_params respects disabled modifier"""
        from ai_utils import ReasoningLevel
        
        mock_bot = MagicMock()
        mock_bot.agent_model_name = "gpt-5-mini"
        mock_bot.agent_reasoning_level = ReasoningLevel.NONE
        mock_bot.event_logger = MagicMock()
        
        # Create controller with modifier disabled
        controller = AgentController(
            mock_bot,
            act_enable_target_context_guard=True,
            act_enable_modifier=False,
            act_enable_additional_context=True
        )
        
        # Test with an action that would normally create a modifier
        params = controller._parse_action_for_act_params("click: second button", "test prompt")
        
        # modifier should be None even though "second" was specified
        assert params["modifier"] is None
    
    def test_extract_act_params_respects_additional_context_disabled(self):
        """Test that _parse_action_for_act_params respects disabled additional_context"""
        from ai_utils import ReasoningLevel
        
        mock_bot = MagicMock()
        mock_bot.agent_model_name = "gpt-5-mini"
        mock_bot.agent_reasoning_level = ReasoningLevel.NONE
        mock_bot.event_logger = MagicMock()
        
        # Create controller with additional_context disabled
        controller = AgentController(
            mock_bot,
            act_enable_target_context_guard=True,
            act_enable_modifier=True,
            act_enable_additional_context=False
        )
        
        # Test with an action that would normally create additional_context
        params = controller._parse_action_for_act_params("click: first button", "test prompt")
        
        # additional_context should be empty even though "first" and "button" were specified
        assert params["additional_context"] == ""
    
    def test_extract_act_params_default_behavior(self):
        """Test that _parse_action_for_act_params works normally with all enabled"""
        from ai_utils import ReasoningLevel
        
        mock_bot = MagicMock()
        mock_bot.agent_model_name = "gpt-5-mini"
        mock_bot.agent_reasoning_level = ReasoningLevel.NONE
        mock_bot.event_logger = MagicMock()
        
        # Create controller with all enabled (default)
        controller = AgentController(
            mock_bot,
            act_enable_target_context_guard=True,
            act_enable_modifier=True,
            act_enable_additional_context=True
        )
        
        # Test with an action that uses ordinal selection
        params = controller._parse_action_for_act_params("click: first button", "test prompt")
        
        # All parameters should be populated
        assert params["modifier"] is not None
        assert params["target_context_guard"] is not None
        assert params["additional_context"] != ""
        
        # Verify the actual values
        assert "ordinal:0" in params["modifier"]
        assert "first" in params["target_context_guard"].lower()
        assert "button" in params["additional_context"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])




